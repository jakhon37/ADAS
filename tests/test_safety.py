"""Tests for SafetyMonitor: the advisory API and the limits it enforces.

Every field of :class:`adas.control.safety.SafetyLimits` has a test that can trip
it -- the ones the monitor checks directly here, the ones the arbiter checks in
``tests/test_arbiter.py``. A limit that no test can trip is a limit that is not
implemented, which was the state of ``max_steering_rate_rad_s`` and
``max_lateral_offset_m`` before this workstream.

The authoritative arbitration path is exercised in ``tests/test_arbiter.py``.
"""

from __future__ import annotations

import pytest

from adas.control import SafetyLimits, SafetyMonitor
from adas.control.arbiter import SafetyContext
from adas.core.exceptions import SafetyViolation
from adas.core.models import (
    BoundingBox,
    ControlCommand,
    EgoState,
    MotionPlan,
    SafetyState,
    TrackedObject,
)

DT = 0.05


def _lead(distance_m: float, velocity_mps: float = 0.0) -> TrackedObject:
    """``velocity_mps`` keeps the tracker's positive-is-closing convention."""
    return TrackedObject(
        track_id=1,
        box=BoundingBox(x1=600, y1=300, x2=700, y2=460, confidence=0.9, label="car"),
        velocity_mps=velocity_mps,
        distance_m=distance_m,
    )


# --------------------------------------------------------------------------- #
# check_motion_plan
# --------------------------------------------------------------------------- #


def test_speed_limit_is_enforced():
    monitor = SafetyMonitor()
    with pytest.raises(SafetyViolation, match="exceeds limit"):
        monitor.check_motion_plan(MotionPlan(50.0, 0.0, "test"), current_speed_mps=10.0)


def test_steering_angle_limit_is_enforced():
    monitor = SafetyMonitor()
    with pytest.raises(SafetyViolation, match="Steering angle"):
        monitor.check_motion_plan(MotionPlan(10.0, 40.0, "test"), current_speed_mps=10.0)


def test_requested_acceleration_limit_is_enforced():
    """Acceleration is the energy-adding direction and must be the branch that raises."""
    monitor = SafetyMonitor(SafetyLimits(max_acceleration_mps2=3.0, plan_horizon_s=1.0))
    with pytest.raises(SafetyViolation, match="Requested acceleration"):
        monitor.check_motion_plan(MotionPlan(20.0, 0.0, "test"), current_speed_mps=10.0)


def test_hard_braking_is_never_a_plan_violation():
    """The old monitor vetoed the only correct action in an emergency.

    Ego at 20 m/s with a stopped car at 3 m: the plan asks for a near-stop, which
    is a 16 m/s^2 request over a 1 s horizon. That must not raise.
    """
    monitor = SafetyMonitor()
    monitor.check_motion_plan(MotionPlan(3.75, 0.0, "aeb"), current_speed_mps=20.0)


def test_valid_plan_passes():
    monitor = SafetyMonitor()
    monitor.check_motion_plan(MotionPlan(15.0, 10.0, "test"), current_speed_mps=13.0)
    assert monitor.advisory_violations == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_plan_fields_raise(value):
    monitor = SafetyMonitor()
    with pytest.raises(SafetyViolation):
        monitor.check_motion_plan(MotionPlan(value, 0.0, "bad"), current_speed_mps=10.0)
    with pytest.raises(SafetyViolation):
        monitor.check_motion_plan(MotionPlan(10.0, value, "bad"), current_speed_mps=10.0)


# --------------------------------------------------------------------------- #
# check_following_distance -- ADAS-DEC-15
# --------------------------------------------------------------------------- #


def test_following_distance_uses_an_rss_minimum_not_a_fixed_2m():
    monitor = SafetyMonitor()
    # 25 m behind a stopped car at 20 m/s: stopping distance alone is 33 m.
    with pytest.raises(SafetyViolation, match="Following distance"):
        monitor.check_following_distance(_lead(25.0, velocity_mps=20.0), ego_speed_mps=20.0)


def test_following_distance_accounts_for_the_lead_speed():
    """Same gap, same ego speed, but a lead moving with us is not a violation."""
    monitor = SafetyMonitor()
    monitor.check_following_distance(_lead(25.0, velocity_mps=0.0), ego_speed_mps=20.0)


def test_following_distance_absolute_floor():
    monitor = SafetyMonitor()
    with pytest.raises(SafetyViolation, match="Following distance"):
        monitor.check_following_distance(_lead(1.0), ego_speed_mps=0.0)


def test_no_lead_vehicle_is_not_a_violation():
    monitor = SafetyMonitor()
    monitor.check_following_distance(None, ego_speed_mps=25.0)


def test_non_finite_lead_range_raises():
    monitor = SafetyMonitor()
    with pytest.raises(SafetyViolation):
        monitor.check_following_distance(_lead(float("nan")), ego_speed_mps=10.0)


# --------------------------------------------------------------------------- #
# check_control_command -- ADAS-DEC-14
# --------------------------------------------------------------------------- #


def test_steering_rate_is_actually_limited_now():
    """The old implementation computed the rate into a local and discarded it."""
    limits = SafetyLimits(max_steering_rate_rad_s=0.5, max_road_wheel_rad=0.436)
    monitor = SafetyMonitor(limits)
    out = monitor.check_control_command(ControlCommand(0.0, 0.0, 1.0), dt=DT)
    expected = (limits.max_steering_rate_rad_s * DT) / limits.max_road_wheel_rad
    assert out.steering == pytest.approx(expected, rel=1e-6)
    assert monitor.advisory_violations == 1


def test_steering_rate_limit_applies_on_the_very_first_command():
    """The old guard ``if self._last_timestamp > 0`` disabled the first frame."""
    monitor = SafetyMonitor()
    out = monitor.check_control_command(ControlCommand(0.0, 0.0, -1.0), dt=DT)
    assert abs(out.steering) < 1.0


def test_steering_within_the_rate_limit_passes_through():
    monitor = SafetyMonitor()
    out = monitor.check_control_command(ControlCommand(0.0, 0.0, 0.01), dt=DT)
    assert out.steering == pytest.approx(0.01)
    assert monitor.advisory_violations == 0


def test_non_finite_command_field_raises():
    monitor = SafetyMonitor()
    with pytest.raises(SafetyViolation):
        monitor.check_control_command(ControlCommand(0.0, float("nan"), 0.0), dt=DT)


# --------------------------------------------------------------------------- #
# sanitize_control_command
# --------------------------------------------------------------------------- #


def test_command_sanitization_clamps_ranges():
    monitor = SafetyMonitor()
    out = monitor.sanitize_control_command(ControlCommand(throttle=1.5, brake=-0.5, steering=2.0))
    assert out.throttle == 1.0
    assert out.brake == 0.0
    assert out.steering == 1.0


def test_negative_throttle_is_clamped_to_zero_not_to_minus_one():
    monitor = SafetyMonitor()
    assert monitor.sanitize_control_command(ControlCommand(-0.7, 0.0, 0.0)).throttle == 0.0


def test_nan_brake_does_not_become_full_braking():
    """``max(0.0, min(1.0, nan))`` is 1.0 in Python; the old clamp shipped that."""
    monitor = SafetyMonitor()
    out = monitor.sanitize_control_command(ControlCommand(0.0, float("nan"), 0.0))
    assert out.brake == 0.0
    assert out.throttle == 0.0
    assert out.steering == 0.0


def test_simultaneous_throttle_and_brake_drops_the_throttle():
    monitor = SafetyMonitor()
    out = monitor.sanitize_control_command(ControlCommand(0.8, 0.5, 0.0))
    assert out.throttle == 0.0
    assert out.brake == 0.5


# --------------------------------------------------------------------------- #
# The authoritative path is reachable from the monitor
# --------------------------------------------------------------------------- #


def test_monitor_exposes_the_arbiter():
    monitor = SafetyMonitor()
    context = SafetyContext(ego=EgoState(speed_mps=10.0, valid=True), dt_s=DT)
    result = monitor.arbitrate(MotionPlan(11.0, 0.0, "ok"), ControlCommand(0.2, 0.0, 0.0), context)
    assert result.state is SafetyState.NOMINAL
    assert result.command.throttle == pytest.approx(0.2)
    assert monitor.state is SafetyState.NOMINAL


def test_monitor_reset_clears_the_arbiter_latch():
    monitor = SafetyMonitor()
    context = SafetyContext(ego=None, dt_s=DT)
    monitor.arbitrate(MotionPlan(11.0, 0.0, "ok"), ControlCommand(1.0, 0.0, 0.0), context)
    assert monitor.state is not SafetyState.NOMINAL
    monitor.reset()
    assert monitor.state is SafetyState.NOMINAL
    assert monitor.advisory_violations == 0


def test_limits_project_onto_the_arbiter():
    limits = SafetyLimits(
        max_speed_mps=20.0, max_steering_rate_rad_s=0.25, max_lateral_offset_m=0.9
    )
    arbiter_limits = limits.to_arbiter_limits()
    assert arbiter_limits.max_speed_mps == 20.0
    assert arbiter_limits.max_steering_rate_rad_s == 0.25
    assert arbiter_limits.max_lateral_offset_m == 0.9
    assert arbiter_limits.standstill_gap_m >= arbiter_limits.absolute_min_gap_m


def test_every_arbiter_limit_is_reachable_from_safety_limits():
    """``to_arbiter_limits`` silently dropped whole groups of limits.

    ``range_disagreement_frac``, ``min_dt_s``, ``max_dt_s``, ``max_frame_gap_s``,
    the entire ``aeb_*`` group, ``kinematics_min_speed_mps``,
    ``limited_throttle_cap`` and the output-shaping rates existed only as
    ``ArbiterLimits`` defaults, so two of the thresholds that can latch the TERMINAL
    DISENGAGE state could not be changed for a deployment without editing source.
    This test fails the moment a new arbiter limit is added without a projection.
    """
    from dataclasses import fields

    projected = SafetyLimits().to_arbiter_limits()
    distinctive = {
        "range_disagreement_frac": 0.11,
        "min_range_confidence": 0.12,
        "range_corroboration_frames": 7,
        "min_dt_s": 0.013,
        "max_dt_s": 0.77,
        "max_frame_gap_s": 0.66,
        "aeb_required_decel_mps2": 5.5,
        "aeb_decel_margin": 1.05,
        "aeb_min_decel_mps2": 3.7,
        "aeb_headway_frac": 0.42,
        "kinematics_min_speed_mps": 0.9,
        "limited_throttle_cap": 0.15,
        "throttle_rate_per_s": 3.3,
        "brake_release_rate_per_s": 6.6,
        "min_in_path_half_width_frac": 0.44,
        "lane_trust_confidence": 0.66,
        "mrm_straighten_speed_mps": 2.2,
        "allow_uncalibrated_range": True,
    }
    tuned = SafetyLimits(**distinctive).to_arbiter_limits()
    for name, value in distinctive.items():
        assert getattr(tuned, name) == value, "%s is not projected" % name

    unreachable = []
    for field_info in fields(type(projected)):
        name = field_info.name
        if name in distinctive:
            continue
        if not hasattr(SafetyLimits, name) and name not in (
            # Projected under a different SafetyLimits name.
            "absolute_min_gap_m",
            "max_acceleration_mps2",
            "max_deceleration_mps2",
            "plan_horizon_s",
        ):
            unreachable.append(name)
    assert unreachable == [], (
        "ArbiterLimits fields with no SafetyLimits counterpart: %r" % unreachable
    )
