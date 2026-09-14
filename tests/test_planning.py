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
    """The ACCELERATION limit lives in the controller, and this proves it holds.

    The planner used to rate-limit its own target speed, which made the target a
    trajectory rather than a setpoint and was the mechanism by which "brake at
    3 m/s^2" reached the controller as a 0.15 m/s step it could not answer.  The
    target is now the setpoint the driver asked for and the limit on how fast the
    vehicle may approach it is enforced where the actuator is, so the property is
    tested where it now lives: through the pedal.
    """
    from adas.control import PIDLikeLongitudinalController

    planner = BehaviorPlanner(cruise_speed_mps=15.0, max_accel_mps2=2.0)
    controller = PIDLikeLongitudinalController()
    speed = 0.0
    previous_accel = 0.0
    for _ in range(600):
        plan = planner.plan(WIDTH, WIDTH / 2.0, [], ego=_ego(speed), dt_s=DT)
        assert plan.target_speed_mps <= 15.0 + 1e-9
        cmd = controller.to_command(plan, speed, dt_s=DT)
        accel = (
            cmd.throttle * controller.accel_authority_mps2
            - cmd.brake * controller.brake_authority_mps2
        )
        assert accel <= controller.accel_authority_mps2 + 1e-9
        assert abs(accel - previous_accel) <= controller.max_jerk_mps3 * DT + 1e-9
        previous_accel = accel
        speed = max(0.0, speed + accel * DT)
    assert speed == pytest.approx(15.0, abs=0.2)


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


def test_the_planner_does_not_read_the_trackers_rate_channel_at_all():
    """``TrackedObject.velocity_mps`` is not an input to the decision any more.

    It is an unfiltered reciprocal derivative computed by the tracker from the
    same range the planner is given, so believing it is believing one channel
    twice.  The planner differentiates the range itself and carries the standard
    error of doing so.  This pins the property directly: a rate channel that
    lies, in either direction, changes nothing.
    """
    def run(reported_rate):
        planner = BehaviorPlanner(ego_lane_half_width_frac=0.25)
        out = []
        gap = 45.0
        for index in range(12):
            track = _track(track_id=1, distance_m=gap, velocity_mps=reported_rate)
            track.hits = index + 1
            out.append(planner.plan(
                WIDTH, WIDTH / 2.0, [track], ego=_ego(12.0), dt_s=DT
            ))
            gap -= 8.0 * DT
        return out

    honest = run(+8.0)
    lying = run(-8.0)
    absent = run(0.0)
    for a, b, c in zip(honest, lying, absent):
        assert a.decel_demand_mps2 == pytest.approx(b.decel_demand_mps2, abs=1e-12)
        assert a.decel_demand_mps2 == pytest.approx(c.decel_demand_mps2, abs=1e-12)
        assert a.target_speed_mps == pytest.approx(b.target_speed_mps, abs=1e-12)


def test_a_measured_closure_lowers_the_target_below_a_steady_gap():
    """The sign of the planner's OWN estimate, end to end through lead selection."""
    def run(closing):
        planner = BehaviorPlanner(ego_lane_half_width_frac=0.25)
        gap = 45.0
        decision = None
        for index in range(12):
            track = _track(track_id=1, distance_m=gap)
            track.hits = index + 1
            decision = planner.plan(WIDTH, WIDTH / 2.0, [track], ego=_ego(12.0), dt_s=DT)
            gap -= closing * DT
        return decision

    closing = run(+8.0)
    receding = run(-8.0)
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


def test_the_in_path_gate_is_metric_and_ego_relative():
    """A car in the next lane is not in the way, however the lane is fitted.

    The gate is ``|lateral offset from the EGO| <= half the ego + half the
    object``: 1.8 m for two passenger cars.  A car one lane over is at 3.5 m.
    The offset comes from the detector's own box geometry, which no lane model
    can move -- which is the whole point, because a lane fit that has slipped
    0.86 m drags a 1.8 m car at 3.5 m inside the *reported* ego lane and a system
    that gated on the lane would brake for it.
    """
    planner = BehaviorPlanner(ego_lane_half_width_frac=0.25)
    next_lane = _track(track_id=9, distance_m=12.0, lateral_offset_m=3.5)
    ahead = _track(track_id=1, distance_m=50.0, lateral_offset_m=0.0)
    planner.plan(WIDTH, WIDTH / 2.0, [next_lane, ahead], ego=_ego(12.0), dt_s=DT)
    assert planner.last_speed_decision.lead_track_id == 1

    # And the converse: an object that IS in the way is selected even though it
    # is further away than the one beside the lane.
    planner = BehaviorPlanner(ego_lane_half_width_frac=0.25)
    planner.plan(
        WIDTH, WIDTH / 2.0,
        [_track(track_id=9, distance_m=12.0, lateral_offset_m=3.5),
         _track(track_id=2, distance_m=26.0, lateral_offset_m=0.4)],
        ego=_ego(12.0), dt_s=DT,
    )
    assert planner.last_speed_decision.lead_track_id == 2


def test_the_in_ego_lane_flag_is_never_read():
    """``TrackedObject.in_ego_lane`` is the LANE ESTIMATOR's opinion.

    It is computed by the tracker from the same lane model the planner would
    otherwise use, and it fails in both directions: a lane fit that has slipped
    toward the kerb hides a car directly in front of the bumper, and one that has
    slipped the other way reports a car in the next lane as being in the way.
    The planner used to PREFER it whenever any object carried it, which is how a
    lane error came to create a hazard.  Setting it must now change nothing.
    """
    for flag in (False, True):
        planner = BehaviorPlanner(ego_lane_half_width_frac=0.25)
        beside = _track(track_id=7, distance_m=12.0, lateral_offset_m=3.5, in_ego_lane=flag)
        ahead = _track(track_id=8, distance_m=40.0, lateral_offset_m=0.0, in_ego_lane=not flag)
        planner.plan(WIDTH, WIDTH / 2.0, [beside, ahead], ego=_ego(12.0), dt_s=DT)
        assert planner.last_speed_decision.lead_track_id == 8, (
            "the lane flag moved the in-path gate (flag=%s)" % flag
        )


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
    brakes.  Through an entire cut-in the controller's own brake used to stay at
    0.000 while the arbiter's independent brake went to 1.000.  This test uses NO
    arbiter: the planner and the controller must produce a real emergency brake by
    themselves.

    What has changed since it was written is WHEN, not whether.  The planner no
    longer declares an emergency on the first frame of a brand-new track, because
    on that frame it has one range measurement and no closing rate at all, and
    the only prior available -- "assume it is stationary" -- is the phantom this
    redesign exists to remove.  The requirement is therefore stated as the
    physics states it: emergency-grade braking within the measurement floor plus
    the jerk ramp, and full authority in time to matter.
    """
    from adas.control import PIDLikeLongitudinalController

    planner = BehaviorPlanner(ego_lane_half_width_frac=0.25)
    controller = PIDLikeLongitudinalController()

    speed = 15.0
    gap = 15.0
    brakes = []
    reasons = []
    for index in range(30):
        track = _track(track_id=1, distance_m=gap, velocity_mps=speed, lateral_offset_m=0.0)
        track.age_frames = index + 1
        track.hits = index + 1
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

    fired = next((i for i, r in enumerate(reasons) if "aeb" in r), None)
    assert fired is not None, reasons[:6]
    # Frame 2 is the first with a closing rate at all; the reason only reads
    # "aeb" once the SHAPED demand has reached emergency grade, which the
    # 20 m/s^3 ceiling puts 3.5 frames after that.
    assert fired <= 6, "emergency declared only at frame %d: %s" % (fired, reasons[:6])
    assert max(brakes) >= 0.99, "controller peak brake was only %.3f" % max(brakes)
    full = next((i for i, b in enumerate(brakes) if b >= 0.99), None)
    assert full is not None and full <= fired + 9, (
        "full authority took %d frames after the emergency was declared" % (full - fired)
    )
    # 15 m of gap at 15 m/s is not avoidable, so the claim here is speed
    # reduction.  The arithmetic bounds how much: a 20 m/s^3 ramp to full
    # authority covers 5.79 m and sheds 1.6 m/s, the remaining 9.21 m at
    # 8 m/s^2 leaves 5.7 m/s, and starting at the first frame a closing rate can
    # exist rather than at frame 0 costs 1.5 m and takes that to 7.5 m/s.  A
    # system that beat 7.5 m/s here would have braked before it could measure
    # anything, which is the phantom this redesign exists to remove.
    assert speed < 9.0, "impact speed only fell to %.2f m/s" % speed


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


def test_the_plan_carries_a_deceleration_demand_not_a_zero_target():
    """The co-design contract between the planner and the controller.

    A target speed cannot express a deceleration.  ``MotionPlan.decel_demand_mps2``
    can, it is jerk shaped by the planner, and it is the only source of brake in
    the primary path -- which is what makes the primary path able to stop the
    vehicle without the arbiter.
    """
    planner = BehaviorPlanner(ego_lane_half_width_frac=0.25)
    gap = 30.0
    plan = None
    for index in range(14):
        track = _track(track_id=1, distance_m=gap, lateral_offset_m=0.0)
        track.hits = index + 1
        plan = planner.plan(
            WIDTH, WIDTH / 2.0, [track], ego=_ego(15.0), dt_s=DT, frame_height_px=HEIGHT
        )
        gap -= 15.0 * DT
    assert plan.decel_demand_mps2 is not None
    assert plan.decel_demand_mps2 > 3.5, (
        "a 15 m/s closure inside 20 m produced only %.2f m/s^2" % plan.decel_demand_mps2
    )
    assert "aeb" in plan.reason


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


# --------------------------------------------------------------------------- #
# The primary path must be able to stop the car BY ITSELF
#
# Regression for the blocker "the planner+controller still cannot execute an AEB
# on their own; the arbiter is the only thing that brakes hard enough to avoid a
# collision". The arbiter is meant to be an independent backstop; if it is the
# only effective brake there is no redundancy and every arbiter defect is a
# single point of failure. None of these tests instantiate an arbiter.
# --------------------------------------------------------------------------- #

W_PX, H_PX = 1280, 720
ACCEL_AUTHORITY = 2.5
BRAKE_AUTHORITY = 8.0


def _lead_track(gap_m, closing_mps, frame_index=0):
    return TrackedObject(
        track_id=1,
        box=BoundingBox(600.0, 300.0, 680.0, 400.0, 0.9, "car"),
        velocity_mps=closing_mps,
        distance_m=gap_m,
        age_frames=frame_index,
        hits=frame_index,
        time_since_update=0,
        in_ego_lane=True,
    )


def _drive_closed_loop(gap0, ego_v0, lead_v0, lead_accel, frames, dt=DT):
    """Plant driven by the CONTROLLER ONLY. Returns (rows, collided)."""
    from adas.control import PIDLikeLongitudinalController

    planner = BehaviorPlanner(ego_lane_half_width_frac=0.25)
    controller = PIDLikeLongitudinalController()
    v, lead_v, gap, t = ego_v0, lead_v0, gap0, 0.0
    rows = []
    for i in range(frames):
        plan = planner.plan(
            frame_width_px=W_PX,
            lane_center_px=W_PX / 2.0,
            objects=[_lead_track(gap, v - lead_v, i)],
            ego=EgoState(speed_mps=v, valid=True, timestamp_s=t),
            perception_valid=True,
            dt_s=dt,
            frame_height_px=H_PX,
        )
        emergency = "aeb" in plan.reason
        cmd = controller.to_command(plan, v, dt_s=dt, emergency=emergency)
        rows.append((t, gap, v, plan.target_speed_mps, cmd.throttle, cmd.brake, plan.reason))
        v = max(0.0, v + (ACCEL_AUTHORITY * cmd.throttle - BRAKE_AUTHORITY * cmd.brake) * dt)
        lead_v = max(0.0, lead_v + lead_accel(t) * dt)
        gap += (lead_v - v) * dt
        t += dt
        if gap <= 0.0:
            return rows, True
    return rows, False


def test_planner_and_controller_alone_stop_behind_a_decelerating_lead():
    """No arbiter anywhere: the primary path must avoid the collision by itself."""
    rows, collided = _drive_closed_loop(
        gap0=30.0, ego_v0=15.0, lead_v0=15.0, lead_accel=lambda t: -4.0, frames=300
    )
    assert not collided, "collided; min gap %.2f m" % min(r[1] for r in rows)
    assert min(r[1] for r in rows) > 1.0
    assert rows[-1][2] < 0.5, "ego never stopped: v=%.2f" % rows[-1][2]


def test_planner_and_controller_alone_reach_full_brake_in_a_cut_in():
    """The CONTROLLER's own brake column, not the arbiter's, must saturate."""
    rows, _ = _drive_closed_loop(
        gap0=15.0, ego_v0=15.0, lead_v0=0.0, lead_accel=lambda t: 0.0, frames=30
    )
    fired = next((r[0] for r in rows if "aeb" in r[6]), None)
    assert fired is not None, [r[6] for r in rows[:6]]
    # 0.10 s of measurement floor plus the 0.175 s the 20 m/s^3 ceiling takes to
    # build 3.5 m/s^2 from zero.
    assert fired <= 0.30, "emergency declared only at %.2f s" % fired
    full = next((r[0] for r in rows if r[5] >= 0.99), None)
    assert full is not None, "controller never reached full brake: %s" % [
        round(r[5], 3) for r in rows
    ]
    assert full <= fired + 0.45, (
        "full brake %.2f s after the emergency was declared, against the 0.40 s the "
        "20 m/s^3 emergency jerk ceiling allows: %s"
        % (full - fired, [round(r[5], 3) for r in rows])
    )


def test_closed_loop_recovers_to_no_brake_when_the_hazard_clears():
    """Feed-forward is an output that becomes an input; it must not self-sustain."""

    def profile(t):
        if t < 2.0:
            return 0.0
        if t < 5.0:
            return -4.0
        if t < 9.0:
            return 0.0
        if t < 13.0:
            return 2.0
        return 0.0

    rows, collided = _drive_closed_loop(
        gap0=40.0, ego_v0=15.0, lead_v0=15.0, lead_accel=profile, frames=400
    )
    assert not collided
    tail = rows[-60:]
    assert all(r[5] == 0.0 for r in tail), "still braking after recovery: %s" % [
        round(r[5], 3) for r in tail
    ]
    # Converging on the spacing policy's fixed point, not ringing around it.  The
    # test is monotonicity rather than a width, because the lead spends four
    # seconds accelerating away and the gap it leaves behind is genuinely still
    # opening when the run ends -- a window narrow enough to catch an oscillation
    # would fail a correct, slow approach to equilibrium.
    gaps = [r[1] for r in tail]
    turns = sum(
        1
        for i in range(1, len(gaps) - 1)
        if (gaps[i] - gaps[i - 1]) * (gaps[i + 1] - gaps[i]) < -1e-6
    )
    assert turns <= 1, "gap reversed direction %d times in the tail" % turns
    switches = 0
    previous = "coast"
    for r in rows:
        mode = "throttle" if r[4] > 0 else ("brake" if r[5] > 0 else "coast")
        if mode != previous and "coast" not in (mode, previous):
            switches += 1
        previous = mode
    assert switches <= 2, "%d direct throttle<->brake switches in 400 frames" % switches


def test_no_braking_when_the_gap_is_already_at_equilibrium():
    """400 frames at the spacing policy's own fixed point: zero pedal action."""
    rows, collided = _drive_closed_loop(
        gap0=42.0, ego_v0=15.0, lead_v0=15.0, lead_accel=lambda t: 0.0, frames=400
    )
    assert not collided
    assert max(r[5] for r in rows) == 0.0
    assert max(r[4] for r in rows) == 0.0
