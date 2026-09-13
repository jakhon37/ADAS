"""Tests for the behavior planner (composition of the two laws).

The longitudinal law itself is swept in ``tests/test_longitudinal.py``; this file
tests what ``BehaviorPlanner`` adds: lead selection, the ego-lane gate, the sign
conversion out of the tracker, the explicit ``perception_valid`` contract, the
lateral law's speed scheduling and lateral-acceleration cap, and the reason string.
"""

from __future__ import annotations

import logging
import math
import random

import pytest

from adas.core.exceptions import PlanningError
from adas.core.models import (
    BoundingBox,
    EgoState,
    LaneModel,
    RangeEstimate,
    RangeSource,
    TrackedObject,
)
from adas.planning import BehaviorPlanner
from adas.planning.lateral import CameraGeometry, LateralLimits, LateralPlanner

SEED = 20240913
DT = 0.05
WIDTH = 1280
HEIGHT = 720


def _ego(speed_mps: float) -> EgoState:
    return EgoState(speed_mps=speed_mps, valid=True, timestamp_s=0.0)


def _track(
    track_id: int = 1,
    distance_m: float = 30.0,
    velocity_mps: float = 0.0,
    center_x: float = WIDTH / 2.0,
    **kwargs,
) -> TrackedObject:
    """``velocity_mps`` keeps the tracker's positive-is-closing convention."""
    return TrackedObject(
        track_id=track_id,
        box=BoundingBox(
            x1=center_x - 60.0,
            y1=300.0,
            x2=center_x + 60.0,
            y2=460.0,
            confidence=0.9,
            label="car",
        ),
        velocity_mps=velocity_mps,
        distance_m=distance_m,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Longitudinal composition
# --------------------------------------------------------------------------- #


def test_planner_cruises_on_an_empty_road():
    planner = BehaviorPlanner(cruise_speed_mps=15.0)
    speed = 15.0
    for _ in range(200):
        plan = planner.plan(WIDTH, WIDTH / 2.0, [], ego=_ego(speed), dt_s=DT)
        speed = plan.target_speed_mps
    assert plan.target_speed_mps == pytest.approx(15.0)
    assert "cruise" in plan.reason


def test_planner_accelerates_to_cruise_within_the_accel_limit():
    planner = BehaviorPlanner(cruise_speed_mps=15.0, max_accel_mps2=2.0)
    previous = 0.0
    for _ in range(400):
        plan = planner.plan(WIDTH, WIDTH / 2.0, [], ego=_ego(previous), dt_s=DT)
        assert plan.target_speed_mps - previous <= 2.0 * DT + 1e-9
        previous = plan.target_speed_mps
    assert previous == pytest.approx(15.0)


def test_planner_slows_for_a_close_vehicle():
    planner = BehaviorPlanner(cruise_speed_mps=15.0, min_follow_distance_m=12.0)
    plan = planner.plan(WIDTH, WIDTH / 2.0, [_track(distance_m=6.0)], ego=_ego(10.0), dt_s=DT)
    assert plan.target_speed_mps < 10.0
    assert "follow" in plan.reason or "aeb" in plan.reason


def test_planner_stops_behind_a_stationary_vehicle_inside_the_standstill_gap():
    planner = BehaviorPlanner(cruise_speed_mps=15.0, standstill_gap_m=4.0)
    speed = 3.0
    for _ in range(300):
        plan = planner.plan(WIDTH, WIDTH / 2.0, [_track(distance_m=1.0)], ego=_ego(speed), dt_s=DT)
        speed = plan.target_speed_mps
    assert speed == pytest.approx(0.0, abs=1e-9)


def test_planner_target_is_monotone_in_lead_distance():
    """End-to-end monotonicity through lead selection, not just the pure law."""
    previous = -1.0
    for step in range(0, 600):
        distance = step * 0.1
        planner = BehaviorPlanner(cruise_speed_mps=15.0)
        plan = planner.plan(
            WIDTH, WIDTH / 2.0, [_track(distance_m=distance)], ego=_ego(12.0), dt_s=DT
        )
        assert plan.target_speed_mps >= previous - 1e-9, "non-monotone at d=%.1f" % distance
        previous = plan.target_speed_mps


def test_planner_uses_the_tracker_sign_convention_correctly():
    """``TrackedObject.velocity_mps`` is positive-when-closing.

    A closing lead must produce a LOWER target than a receding one at the same
    range. Getting this sign backwards is the failure mode ADAS-DEC-03 warns about.
    """
    closing = BehaviorPlanner().plan(
        WIDTH, WIDTH / 2.0, [_track(distance_m=45.0, velocity_mps=+8.0)], ego=_ego(12.0), dt_s=DT
    )
    receding = BehaviorPlanner().plan(
        WIDTH, WIDTH / 2.0, [_track(distance_m=45.0, velocity_mps=-8.0)], ego=_ego(12.0), dt_s=DT
    )
    assert closing.target_speed_mps < receding.target_speed_mps


def test_planner_selects_the_nearest_in_path_object():
    planner = BehaviorPlanner()
    tracks = [
        _track(track_id=1, distance_m=60.0),
        _track(track_id=2, distance_m=18.0),
        _track(track_id=3, distance_m=95.0),
    ]
    planner.plan(WIDTH, WIDTH / 2.0, tracks, ego=_ego(12.0), dt_s=DT)
    assert planner.last_speed_decision is not None
    assert planner.last_speed_decision.lead_track_id == 2


def test_ego_lane_gate_excludes_objects_outside_the_band():
    planner = BehaviorPlanner(ego_lane_half_width_frac=0.1)
    edge = _track(track_id=9, distance_m=3.0, center_x=60.0)
    ahead = _track(track_id=1, distance_m=50.0)
    planner.plan(WIDTH, WIDTH / 2.0, [edge, ahead], ego=_ego(12.0), dt_s=DT)
    assert planner.last_speed_decision.lead_track_id == 1


def test_in_ego_lane_flag_overrides_the_pixel_band_when_populated():
    planner = BehaviorPlanner(ego_lane_half_width_frac=0.1)
    flagged = _track(track_id=7, distance_m=20.0, center_x=40.0, in_ego_lane=True)
    unflagged = _track(track_id=8, distance_m=5.0)
    planner.plan(WIDTH, WIDTH / 2.0, [flagged, unflagged], ego=_ego(12.0), dt_s=DT)
    assert planner.last_speed_decision.lead_track_id == 7


def test_second_range_channel_is_used_conservatively():
    """When two channels disagree the planner reacts to the nearer one."""
    planner = BehaviorPlanner()
    track = _track(distance_m=40.0)
    track.range_estimate = RangeEstimate(
        distance_m=14.0, confidence=0.8, source=RangeSource.DEPTH_MODEL
    )
    planner.plan(WIDTH, WIDTH / 2.0, [track], ego=_ego(12.0), dt_s=DT)
    assert planner.last_speed_decision.gap_m == pytest.approx(14.0)


def test_object_with_non_finite_range_is_dropped_not_treated_as_infinitely_far():
    planner = BehaviorPlanner()
    bad = _track(track_id=4, distance_m=float("nan"))
    good = _track(track_id=5, distance_m=25.0)
    planner.plan(WIDTH, WIDTH / 2.0, [bad, good], ego=_ego(12.0), dt_s=DT)
    assert planner.last_speed_decision.lead_track_id == 5


# --------------------------------------------------------------------------- #
# Fail-safe contract -- ADAS-DEC-04
# --------------------------------------------------------------------------- #


def test_perception_dropout_is_not_an_empty_road():
    """The regression the whole workstream exists for."""
    planner = BehaviorPlanner(cruise_speed_mps=15.0)
    for _ in range(40):
        planner.plan(WIDTH, WIDTH / 2.0, [], ego=_ego(15.0), dt_s=DT)
    cruising = planner.last_speed_decision.target_speed_mps
    assert cruising == pytest.approx(15.0)

    previous = cruising
    for index in range(1, 80):
        plan = planner.plan(
            WIDTH, WIDTH / 2.0, [], ego=_ego(15.0), perception_valid=False, dt_s=DT
        )
        assert plan.target_speed_mps <= previous + 1e-9
        assert "degraded" in plan.reason
        previous = plan.target_speed_mps
    assert previous < cruising - 5.0


def test_perception_dropout_also_stops_steering():
    planner = BehaviorPlanner()
    plan = planner.plan(WIDTH, 900.0, [], ego=_ego(5.0), perception_valid=False, dt_s=DT)
    assert plan.steering_angle_deg == 0.0
    assert "no_lane" in plan.reason


def test_missing_ego_speed_degrades_rather_than_cruising():
    planner = BehaviorPlanner(cruise_speed_mps=15.0)
    plan = planner.plan(WIDTH, WIDTH / 2.0, [], dt_s=DT)
    assert plan.target_speed_mps == 0.0
    assert "ego_speed_unavailable" in plan.reason


def test_invalid_ego_state_is_ignored():
    planner = BehaviorPlanner()
    invalid = EgoState(speed_mps=12.0, valid=False)
    plan = planner.plan(WIDTH, WIDTH / 2.0, [], ego=invalid, dt_s=DT)
    assert "ego_speed_unavailable" in plan.reason


def test_ego_speed_mps_argument_is_accepted_when_ego_state_is_absent():
    planner = BehaviorPlanner()
    plan = planner.plan(WIDTH, WIDTH / 2.0, [], ego_speed_mps=8.0, dt_s=DT)
    assert plan.target_speed_mps > 0.0
    assert "ego_speed_unavailable" not in plan.reason


def test_invalid_frame_width_raises_planning_error():
    planner = BehaviorPlanner()
    with pytest.raises(PlanningError):
        planner.plan(0, 100.0, [], ego=_ego(5.0), dt_s=DT)


# --------------------------------------------------------------------------- #
# Lateral law -- ADAS-DEC-12
# --------------------------------------------------------------------------- #


def test_lane_centering_direction():
    planner = BehaviorPlanner(max_steering_deg=22.0)
    right = planner.plan(WIDTH, 800.0, [], ego=_ego(0.0), dt_s=DT)
    assert right.steering_angle_deg > 0
    left = planner.plan(WIDTH, 400.0, [], ego=_ego(0.0), dt_s=DT)
    assert left.steering_angle_deg < 0


def test_no_lane_goes_straight():
    planner = BehaviorPlanner()
    plan = planner.plan(WIDTH, None, [], ego=_ego(5.0), dt_s=DT)
    assert plan.steering_angle_deg == 0.0
    assert "no_lane" in plan.reason


def test_lane_center_outside_the_frame_goes_straight():
    planner = BehaviorPlanner()
    plan = planner.plan(WIDTH, WIDTH + 200.0, [], ego=_ego(5.0), dt_s=DT)
    assert plan.steering_angle_deg == 0.0
    assert "invalid_lane" in plan.reason


def test_steering_never_exceeds_the_configured_maximum():
    planner = BehaviorPlanner(max_steering_deg=22.0)
    plan = planner.plan(WIDTH, float(WIDTH), [], ego=_ego(0.0), dt_s=DT)
    assert abs(plan.steering_angle_deg) <= 22.0


def test_steering_is_speed_scheduled():
    """The same pixel error must command less angle at higher speed."""
    slow = BehaviorPlanner().plan(WIDTH, 900.0, [], ego=_ego(2.0), dt_s=DT)
    fast = BehaviorPlanner().plan(WIDTH, 900.0, [], ego=_ego(30.0), dt_s=DT)
    assert abs(fast.steering_angle_deg) < abs(slow.steering_angle_deg)


def test_lateral_acceleration_cap_holds_across_the_speed_range():
    """A 100 px error at 33 m/s used to command a 2.4 g manoeuvre."""
    limits = LateralLimits(max_lateral_accel_mps2=3.0, wheelbase_m=2.8)
    lateral = LateralPlanner(limits)
    for step in range(0, 70):
        speed = step * 0.5
        decision = lateral.plan(WIDTH, float(WIDTH), speed)
        achieved = abs(
            (speed ** 2) * math.tan(math.radians(decision.steering_angle_deg)) / limits.wheelbase_m
        )
        assert achieved <= limits.max_lateral_accel_mps2 + 1e-6, (
            "a_lat %.2f m/s^2 at %.1f m/s" % (achieved, speed)
        )


def test_unknown_ego_speed_uses_the_most_restrictive_schedule():
    lateral = LateralPlanner(LateralLimits(assumed_speed_when_unknown_mps=33.0))
    unknown = lateral.plan(WIDTH, float(WIDTH), None)
    at_33 = lateral.plan(WIDTH, float(WIDTH), 33.0)
    assert unknown.steering_angle_deg == pytest.approx(at_33.steering_angle_deg)


def test_non_metric_path_is_labelled_as_such():
    decision = LateralPlanner().plan(WIDTH, 800.0, 10.0)
    assert decision.is_metric is False
    assert decision.lateral_error_m is None


def test_metric_path_activates_with_camera_extrinsics():
    camera = CameraGeometry(
        focal_length_px=910.0,
        principal_x_px=WIDTH / 2.0,
        principal_y_px=HEIGHT / 2.0,
        mount_height_m=1.3,
        pitch_rad=0.12,
    )
    lateral = LateralPlanner(LateralLimits(), camera=camera)
    decision = lateral.plan(WIDTH, 800.0, 10.0, frame_height_px=HEIGHT)
    assert decision.is_metric is True
    assert decision.lateral_error_m is not None
    assert decision.lateral_error_m > 0.0
    mirrored = lateral.plan(WIDTH, 480.0, 10.0, frame_height_px=HEIGHT)
    assert mirrored.lateral_error_m < 0.0


def test_uncalibrated_camera_config_does_not_enable_the_metric_law():
    """An assumed mount height must not become a metric measurement."""

    class _Config:
        fx, cx, cy = 910.0, 640.0, 360.0
        mount_height_m = 1.30
        pitch_deg = 2.0
        calibrated = False
        label = "uncalibrated-default"

    assert CameraGeometry.from_camera_config(_Config()).is_configured() is False
    opted_in = CameraGeometry.from_camera_config(_Config(), allow_uncalibrated=True)
    assert opted_in.is_configured() is True
    assert opted_in.mount_height_m == pytest.approx(1.30)
    assert opted_in.pitch_rad == pytest.approx(math.radians(2.0))


def test_calibrated_camera_config_enables_the_metric_law():
    class _Config:
        fx, cx, cy = 910.0, 640.0, 360.0
        mount_height_m = 1.30
        pitch_deg = 2.0
        calibrated = True
        label = "bench-2026-09"

    geometry = CameraGeometry.from_camera_config(_Config())
    assert geometry.is_configured() is True
    decision = LateralPlanner(LateralLimits(), camera=geometry).plan(
        WIDTH, 800.0, 10.0, frame_height_px=HEIGHT
    )
    assert decision.is_metric is True


def test_camera_geometry_is_not_configured_by_default():
    assert CameraGeometry().is_configured() is False
    assert CameraGeometry().ground_point_m(100.0, 700.0) is None


def test_ground_point_above_the_horizon_returns_none():
    camera = CameraGeometry(
        focal_length_px=910.0,
        principal_x_px=WIDTH / 2.0,
        principal_y_px=HEIGHT / 2.0,
        mount_height_m=1.3,
        pitch_rad=0.12,
    )
    assert camera.ground_point_m(WIDTH / 2.0, 0.0) is None


# --------------------------------------------------------------------------- #
# Envelope sweep
# --------------------------------------------------------------------------- #


def test_plan_stays_inside_the_envelope_over_a_random_walk():
    rng = random.Random(SEED)
    planner = BehaviorPlanner(cruise_speed_mps=15.0, max_steering_deg=22.0)
    speed = 10.0
    for _ in range(3000):
        tracks = [
            _track(
                track_id=index,
                distance_m=rng.uniform(0.5, 90.0),
                velocity_mps=rng.uniform(-10.0, 20.0),
                center_x=rng.uniform(0.0, WIDTH),
            )
            for index in range(rng.randint(0, 3))
        ]
        lane_center = rng.choice([None, rng.uniform(0.0, WIDTH)])
        valid = rng.random() > 0.15
        plan = planner.plan(
            WIDTH,
            lane_center,
            tracks,
            ego=_ego(speed),
            perception_valid=valid,
            dt_s=rng.uniform(0.02, 0.15),
            lane=LaneModel((0, 0, 0), (0, 0, 0), WIDTH / 2.0, 200.0),
        )
        assert math.isfinite(plan.target_speed_mps)
        assert 0.0 <= plan.target_speed_mps <= 15.0
        assert abs(plan.steering_angle_deg) <= 22.0
        speed = plan.target_speed_mps


def test_reset_clears_planner_state():
    planner = BehaviorPlanner()
    planner.plan(WIDTH, WIDTH / 2.0, [], ego=_ego(12.0), dt_s=DT)
    assert planner.last_speed_decision is not None
    planner.reset()
    assert planner.last_speed_decision is None
    assert planner.longitudinal.previous_target_mps is None


# --------------------------------------------------------------------------- #
# The AEB decision must be EXECUTABLE by the planner + controller path alone
# --------------------------------------------------------------------------- #


def test_planner_and_controller_alone_deliver_a_full_emergency_brake():
    """Regression for the cut-in the safety review found.

    The arbiter is meant to be a backstop, not the only thing in the system that
    brakes.  Through an entire cut-in with ``plan.reason`` reading ``aeb_...`` from
    frame 0, the controller's own brake used to stay at 0.000 while the arbiter's
    independent brake went to 1.000.  This test uses NO arbiter: the planner and
    the controller must produce a real emergency brake by themselves.
    """
    from adas.control import PIDLikeLongitudinalController

    planner = BehaviorPlanner(ego_lane_half_width_frac=0.25)
    controller = PIDLikeLongitudinalController()

    speed = 15.0
    gap = 15.0
    brakes = []
    reasons = []
    for index in range(30):
        track = _track(track_id=1, distance_m=gap, velocity_mps=speed)
        track.in_ego_lane = True
        track.age_frames = index
        track.hits = index
        plan = planner.plan(
            WIDTH,
            WIDTH / 2.0,
            [track],
            ego=_ego(speed),
            perception_valid=True,
            dt_s=DT,
            frame_height_px=HEIGHT,
        )
        reasons.append(plan.reason)
        emergency = "aeb" in plan.reason
        cmd = controller.to_command(plan, speed, dt_s=DT, emergency=emergency)
        brakes.append(cmd.brake)
        assert cmd.throttle == 0.0
        # Integrate the CONTROLLER's own command, not the arbiter's.
        accel = (
            cmd.throttle * controller.accel_authority_mps2
            - cmd.brake * controller.brake_authority_mps2
        )
        speed = max(0.0, speed + accel * DT)
        gap -= speed * DT
        if gap <= 0.0:
            break

    assert all("aeb" in reason for reason in reasons), reasons[:3]
    assert brakes[0] > 0.0, "no brake at all on the first AEB frame"
    assert max(brakes) >= 0.99, "controller peak brake was only %.3f" % max(brakes)
    # 15 m of gap at 15 m/s closing is not avoidable at 8 m/s^2 (14.1 m of pure
    # stopping distance plus the jerk ramp), so the claim here is speed reduction,
    # not avoidance: this run ends at 6.3 m/s where the un-braked case is 15 m/s.
    assert speed < 7.0, "impact speed only fell to %.2f m/s" % speed


def test_an_avoidable_aeb_case_is_actually_avoided_without_the_arbiter():
    """Same path, a gap the geometry does allow: it must stop short."""
    from adas.control import PIDLikeLongitudinalController

    planner = BehaviorPlanner(ego_lane_half_width_frac=0.25)
    controller = PIDLikeLongitudinalController()

    speed = 15.0
    gap = 25.0
    fired = False
    for index in range(200):
        track = _track(track_id=1, distance_m=gap, velocity_mps=speed)
        track.in_ego_lane = True
        plan = planner.plan(
            WIDTH,
            WIDTH / 2.0,
            [track],
            ego=_ego(speed),
            perception_valid=True,
            dt_s=DT,
            frame_height_px=HEIGHT,
        )
        emergency = "aeb" in plan.reason
        fired = fired or emergency
        cmd = controller.to_command(plan, speed, dt_s=DT, emergency=emergency)
        accel = (
            cmd.throttle * controller.accel_authority_mps2
            - cmd.brake * controller.brake_authority_mps2
        )
        speed = max(0.0, speed + accel * DT)
        gap -= speed * DT
        assert gap > 0.0, "collided at frame %d" % index
        if speed <= 0.05:
            break
    assert fired, "AEB never fired; the scenario does not test what it claims"
    assert speed <= 0.05, "never came to a stop (v = %.3f m/s)" % speed
    assert gap > 0.0


def test_aeb_plan_target_is_zero_not_a_trailing_setpoint():
    planner = BehaviorPlanner(ego_lane_half_width_frac=0.25)
    track = _track(track_id=1, distance_m=60.0, velocity_mps=0.0)
    track.in_ego_lane = True
    planner.plan(WIDTH, WIDTH / 2.0, [track], ego=_ego(15.0), dt_s=DT, frame_height_px=HEIGHT)

    close = _track(track_id=1, distance_m=15.0, velocity_mps=15.0)
    close.in_ego_lane = True
    plan = planner.plan(
        WIDTH, WIDTH / 2.0, [close], ego=_ego(15.0), dt_s=DT, frame_height_px=HEIGHT
    )
    assert "aeb" in plan.reason
    assert plan.target_speed_mps == 0.0


# --------------------------------------------------------------------------- #
# Log volume for latched degraded conditions
# --------------------------------------------------------------------------- #


def test_missing_ego_speed_does_not_log_once_per_frame(caplog):
    """ego.source='none' is the shipped default on a board with no CAN bus."""
    planner = BehaviorPlanner()
    with caplog.at_level(logging.DEBUG, logger="adas.planning"):
        for _ in range(1200):
            plan = planner.plan(WIDTH, WIDTH / 2.0, [], ego=None, dt_s=DT)
            assert "ego_speed_unavailable" in plan.reason
            assert planner.ego_speed_available is False
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, "expected 1 WARNING for 1200 frames, got %d" % len(warnings)


def test_an_ego_state_that_lies_about_validity_logs_once(caplog):
    planner = BehaviorPlanner()
    bad = EgoState(speed_mps=float("nan"), valid=True, timestamp_s=0.0)
    with caplog.at_level(logging.DEBUG, logger="adas.planning.behavior_planner"):
        for _ in range(500):
            planner.plan(WIDTH, WIDTH / 2.0, [], ego=bad, dt_s=DT)
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "marked valid" in r.getMessage()
    ]
    assert len(warnings) == 1, "expected 1 WARNING for 500 frames, got %d" % len(warnings)


def test_a_permanently_bad_range_logs_once_not_per_frame(caplog):
    planner = BehaviorPlanner()
    with caplog.at_level(logging.DEBUG, logger="adas.planning.behavior_planner"):
        for _ in range(500):
            broken = _track(track_id=7, distance_m=float("nan"))
            broken.in_ego_lane = True
            planner.plan(WIDTH, WIDTH / 2.0, [broken], ego=_ego(10.0), dt_s=DT)
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "implausible range" in r.getMessage()
    ]
    assert len(warnings) == 1, "expected 1 WARNING for 500 frames, got %d" % len(warnings)


def test_an_out_of_frame_lane_centre_logs_once_not_per_frame(caplog):
    planner = BehaviorPlanner()
    with caplog.at_level(logging.DEBUG, logger="adas.planning.lateral"):
        for _ in range(500):
            planner.plan(WIDTH, float(WIDTH + 500), [], ego=_ego(10.0), dt_s=DT)
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "outside frame" in r.getMessage()
    ]
    assert len(warnings) == 1, "expected 1 WARNING for 500 frames, got %d" % len(warnings)


def test_planner_reset_re_arms_the_log_gates(caplog):
    planner = BehaviorPlanner()
    with caplog.at_level(logging.DEBUG, logger="adas.planning"):
        for _ in range(50):
            planner.plan(WIDTH, WIDTH / 2.0, [], ego=None, dt_s=DT)
        planner.reset()
        assert planner.ego_speed_available is True
        for _ in range(50):
            planner.plan(WIDTH, WIDTH / 2.0, [], ego=None, dt_s=DT)
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "no usable ego speed" in r.getMessage()
    ]
    assert len(warnings) == 2
