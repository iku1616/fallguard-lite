import math

import pytest

from app.fall_detector import FallDetector, FallState, FallThresholds, PoseObservation

UPRIGHT = (100.0, 100.0, 180.0, 300.0)
CROUCHED = (100.0, 190.0, 210.0, 310.0)
FALLING = (100.0, 210.0, 200.0, 360.0)
HORIZONTAL = (80.0, 270.0, 300.0, 370.0)


def obs(track_id, timestamp, bbox, angle=5.0, confidence=0.9):
    return PoseObservation(track_id, timestamp, bbox, angle, confidence)


def establish_standing(detector, track_id="p1", start=0.0):
    result = None
    for offset in (0.0, 0.35, 0.7, 1.05):
        result = detector.update(obs(track_id, start + offset, UPRIGHT))
    assert result is not None
    assert result.metrics["standing_ready"] is True
    return result


def confirm_fall(detector, track_id="p1", start=0.0):
    establish_standing(detector, track_id, start)
    detector.update(obs(track_id, start + 1.25, FALLING, 48.0))
    suspected = detector.update(obs(track_id, start + 1.45, HORIZONTAL, 78.0))
    fallen = detector.update(obs(track_id, start + 2.25, HORIZONTAL, 82.0))
    return suspected, fallen


def test_stable_upright_person_never_alerts():
    detector = FallDetector()
    results = [
        detector.update(obs("p1", timestamp, UPRIGHT))
        for timestamp in (0.0, 0.3, 0.7, 1.1, 1.8, 2.6)
    ]
    assert {result.state for result in results} == {FallState.NORMAL}
    assert all(not result.fall_detected for result in results)


def test_person_already_lying_at_start_is_suppressed():
    detector = FallDetector()
    results = [
        detector.update(obs("p1", timestamp, HORIZONTAL, 85.0))
        for timestamp in (0.0, 0.2, 0.8, 1.6, 3.0)
    ]
    assert all(result.state is FallState.NORMAL for result in results)
    assert all("awaiting_standing_history" in result.reasons for result in results)


def test_fast_drop_then_sustained_horizontal_posture_confirms_fall():
    detector = FallDetector()
    suspected, fallen = confirm_fall(detector)
    assert suspected.state is FallState.SUSPECTED
    assert "rapid_downward_motion" in suspected.reasons
    assert fallen.state is FallState.FALLEN
    assert fallen.fall_detected
    assert "fall_confirmed" in fallen.reasons
    assert fallen.score >= 0.9


def test_suspicion_clears_if_horizontal_evidence_disappears():
    detector = FallDetector()
    establish_standing(detector)
    detector.update(obs("p1", 1.25, FALLING, 50.0))
    assert detector.update(obs("p1", 1.45, HORIZONTAL, 80.0)).state is FallState.SUSPECTED
    cleared = detector.update(obs("p1", 1.7, UPRIGHT, 4.0))
    assert cleared.state is FallState.NORMAL
    assert "suspicion_cleared" in cleared.reasons


def test_slow_deliberate_lying_does_not_trigger():
    detector = FallDetector()
    establish_standing(detector)
    results = [
        detector.update(obs("p1", 2.0, (100.0, 120.0, 190.0, 300.0), 20.0)),
        detector.update(obs("p1", 3.0, (95.0, 170.0, 220.0, 310.0), 40.0)),
        detector.update(obs("p1", 4.0, (80.0, 220.0, 270.0, 330.0), 65.0)),
        detector.update(obs("p1", 5.0, HORIZONTAL, 82.0)),
    ]
    assert all(result.state is FallState.NORMAL for result in results)


def test_crouching_is_not_a_fall():
    detector = FallDetector()
    establish_standing(detector)
    crouch = detector.update(obs("p1", 1.35, CROUCHED, 18.0))
    wait = detector.update(obs("p1", 2.2, CROUCHED, 18.0))
    assert crouch.state is FallState.NORMAL
    assert wait.state is FallState.NORMAL


def test_track_states_do_not_mix_between_people():
    detector = FallDetector()
    establish_standing(detector, "falling")
    establish_standing(detector, "standing")
    detector.update(obs("falling", 1.25, FALLING, 50.0))
    detector.update(obs("falling", 1.45, HORIZONTAL, 80.0))
    fallen = detector.update(obs("falling", 2.25, HORIZONTAL, 80.0))
    standing = detector.update(obs("standing", 2.25, UPRIGHT, 5.0))
    assert fallen.state is FallState.FALLEN
    assert standing.state is FallState.NORMAL


def test_recovery_requires_sustained_upright_posture_and_adds_cooldown():
    detector = FallDetector()
    _, fallen = confirm_fall(detector)
    assert fallen.state is FallState.FALLEN
    recovering = detector.update(obs("p1", 2.45, UPRIGHT, 5.0))
    recovered = detector.update(obs("p1", 3.5, UPRIGHT, 5.0))
    assert recovering.state is FallState.RECOVERING
    assert recovered.state is FallState.NORMAL
    assert "recovery_confirmed" in recovered.reasons
    detector.update(obs("p1", 3.65, FALLING, 50.0))
    ignored = detector.update(obs("p1", 3.8, HORIZONTAL, 80.0))
    assert ignored.state is FallState.NORMAL


def test_missing_and_non_finite_geometry_does_not_corrupt_state():
    detector = FallDetector()
    establish_standing(detector)
    missing = detector.update(obs("p1", 1.2, None, None, None))
    nan_bbox = detector.update(obs("p1", 1.3, (0.0, 0.0, math.nan, 4.0)))
    assert missing.state is FallState.NORMAL
    assert nan_bbox.state is FallState.NORMAL
    assert missing.reasons == ("invalid_bbox",)
    assert nan_bbox.reasons == ("invalid_bbox",)


def test_low_confidence_keypoints_fall_back_to_bbox_evidence():
    detector = FallDetector(FallThresholds(strong_horizontal_aspect_ratio=1.2))
    establish_standing(detector)
    detector.update(obs("p1", 1.25, FALLING, math.nan, 0.9))
    result = detector.update(obs("p1", 1.45, HORIZONTAL, 89.0, 0.05))
    assert result.metrics["torso_angle_degrees"] is None
    assert result.state is FallState.SUSPECTED


def test_non_finite_and_reverse_timestamps_are_rejected():
    detector = FallDetector()
    with pytest.raises(ValueError, match="finite"):
        detector.update(obs("p1", math.nan, UPRIGHT))
    detector.update(obs("p1", 1.0, UPRIGHT))
    with pytest.raises(ValueError, match="monotonic"):
        detector.update(obs("p1", 0.9, UPRIGHT))


def test_ttl_cleanup_expires_only_stale_tracks():
    detector = FallDetector(FallThresholds(track_ttl_seconds=1.0))
    detector.update(obs("stale", 0.0, UPRIGHT))
    detector.update(obs("active", 0.8, UPRIGHT))
    expired = detector.cleanup(1.2)
    assert expired == ("stale",)
    assert detector.active_track_ids == ("active",)


def test_duration_logic_is_seconds_based_with_irregular_intervals():
    detector = FallDetector(FallThresholds(min_standing_seconds=0.9, fallen_hold_seconds=0.6))
    for timestamp in (0.0, 0.11, 0.64, 0.91):
        detector.update(obs("p1", timestamp, UPRIGHT))
    detector.update(obs("p1", 1.02, FALLING, 48.0))
    first = detector.update(obs("p1", 1.16, HORIZONTAL, 80.0))
    too_soon = detector.update(obs("p1", 1.73, HORIZONTAL, 80.0))
    confirmed = detector.update(obs("p1", 1.77, HORIZONTAL, 80.0))
    assert first.state is FallState.SUSPECTED
    assert too_soon.state is FallState.SUSPECTED
    assert confirmed.state is FallState.FALLEN


def test_result_is_json_friendly():
    payload = FallDetector().update(obs(7, 0.0, UPRIGHT)).as_dict()
    assert payload["track_id"] == 7
    assert payload["state"] == "normal"
    assert isinstance(payload["reasons"], list)


def test_invalid_thresholds_fail_fast():
    with pytest.raises(ValueError):
        FallThresholds(track_ttl_seconds=-1.0)
    with pytest.raises(ValueError):
        FallThresholds(history_size=1)
