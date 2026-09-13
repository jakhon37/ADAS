"""Tests for the authoritative safety arbiter.

Three kinds of test live here:

1. **Unit tests** for each envelope check and for the degradation state machine.
2. **Independence tests** proving the arbiter does not inherit the planner's
   beliefs: it ignores ``TrackedObject.velocity_mps``, it re-selects its own lead,
   and it brakes on a hazard even when the plan and the command say "cruise".
3. **Closed-loop scenario tests** driven by :class:`PointMassPlant`, a jerk-limited
   longitudinal plant. Each scenario asserts that the COMMANDED OUTPUT is safe --
   no simulated collision, bounded jerk, no throttle while degraded -- not merely
   that nothing raised.

All randomised tests use a fixed seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional

import pytest

from adas.control import PIDLikeLongitudinalController, SafetyArbiter, SafetyMonitor
from adas.control.arbiter import ArbiterLimits, SafetyContext
from adas.core.models import (
    BoundingBox,
    ControlCommand,
    EgoState,
    LaneModel,
    MotionPlan,
    PerceptionStatus,
    RangeEstimate,
    RangeSource,
    SafetyState,
    TrackedObject,
)
from adas.planning import BehaviorPlanner

SEED = 20240913
DT = 0.05
WIDTH = 1280
HEIGHT = 720


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


def _limits(**kwargs) -> ArbiterLimits:
    return ArbiterLimits(**kwargs)


def _ego(speed_mps: float, valid: bool = True, timestamp_s: float = 0.0) -> EgoState:
    return EgoState(speed_mps=speed_mps, valid=valid, timestamp_s=timestamp_s)


def _track(
    track_id: int = 1,
    distance_m: float = 30.0,
    velocity_mps: float = 0.0,
    center_x: float = WIDTH / 2.0,
    **kwargs,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        box=BoundingBox(
            x1=center_x - 60.0, y1=300.0, x2=center_x + 60.0, y2=460.0,
            confidence=0.9, label="car",
        ),
        velocity_mps=velocity_mps,
        distance_m=distance_m,
        **kwargs,
    )


def _context(
    ego_speed_mps: float = 10.0,
    tracks: Optional[List[TrackedObject]] = None,
    perception_ok: bool = True,
    dt_s: float = DT,
    **kwargs,
) -> SafetyContext:
    return SafetyContext(
        ego=_ego(ego_speed_mps),
        tracks=list(tracks or []),
        perception=PerceptionStatus(ok=perception_ok),
        dt_s=dt_s,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# ADAS-DEC-01: the arbiter is authoritative
# --------------------------------------------------------------------------- #


def test_nominal_command_passes_through_unchanged():
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(11.0, 2.0, "cruise"), ControlCommand(0.25, 0.0, 0.02), _context(10.0)
    )
    assert result.state is SafetyState.NOMINAL
    assert result.violations == []
    assert result.command.throttle == pytest.approx(0.25)
    assert result.command.brake == pytest.approx(0.0)
    assert result.command.steering == pytest.approx(0.02)


def test_over_speed_plan_is_not_actuated():
    """The exact ADAS-DEC-01 scenario: planner asks 40 m/s against a 33 m/s limit."""
    arbiter = SafetyArbiter(_limits(max_speed_mps=33.0))
    result = arbiter.arbitrate(
        MotionPlan(40.0, 0.0, "cruise"), ControlCommand(1.0, 0.0, 0.0), _context(30.0)
    )
    assert result.state is not SafetyState.NOMINAL
    assert result.command.throttle == 0.0
    assert any("plan_speed" in v for v in result.violations)


def test_over_steering_plan_is_not_actuated_at_full_lock():
    """planner 40 deg / controller 25 deg used to reach the actuator as full lock."""
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(10.0, 40.0, "lane"), ControlCommand(0.0, 0.0, 1.0), _context(10.0)
    )
    assert abs(result.command.steering) < 1.0
    assert result.state is not SafetyState.NOMINAL
    assert any("plan_steering" in v for v in result.violations)


def test_result_command_is_never_more_energetic_than_the_input():
    """The arbiter may only ever soften a command.

    Throttle is never raised and BRAKE IS NEVER LOWERED -- not by a frame's worth of
    rate limiting, not at all.  The previous version of this test asserted the
    weaker ``brake >= min(requested, prev + apply_rate*dt)``, which ENCODED the
    defect it should have caught: the arbiter re-applied its own 5.0/s brake
    apply-rate limit against its own previous output, so a handed-in ``brake = 1.0``
    came out 0.250, 0.500, 0.750, 1.000 over four frames while the module docstring
    claimed "brake only ever increased".  Brake jerk shaping, with its emergency
    exemption, belongs to the controller; the arbiter has no business attenuating an
    emergency stop.  Regression guard for that blocker.
    """
    rng = random.Random(SEED)
    arbiter = SafetyArbiter()
    for _ in range(3000):
        cmd = ControlCommand(
            throttle=rng.uniform(0.0, 1.0),
            brake=rng.choice([0.0, rng.uniform(0.0, 1.0)]),
            steering=rng.uniform(-1.0, 1.0),
        )
        if cmd.brake > 0:
            cmd.throttle = 0.0
        result = arbiter.arbitrate(
            MotionPlan(rng.uniform(0.0, 30.0), rng.uniform(-25.0, 25.0), "r"),
            cmd,
            _context(
                ego_speed_mps=rng.uniform(0.0, 30.0),
                tracks=[_track(distance_m=rng.uniform(2.0, 90.0))] if rng.random() < 0.5 else [],
                perception_ok=rng.random() > 0.1,
            ),
        )
        assert result.command.throttle <= cmd.throttle + 1e-9
        assert result.command.brake >= cmd.brake - 1e-9, (
            "the arbiter attenuated a brake: %.3f -> %.3f" % (cmd.brake, result.command.brake)
        )
        assert 0.0 <= result.command.throttle <= 1.0
        assert 0.0 <= result.command.brake <= 1.0
        assert -1.0 <= result.command.steering <= 1.0
        assert not (result.command.throttle > 0.0 and result.command.brake > 0.0)


def test_an_emergency_brake_reaches_the_actuator_on_the_first_frame():
    """The narrow blocker repro (/tmp/rev_brakecut.py), as a permanent test.

    The planner has declared an emergency and the controller has produced
    ``brake = 1.0``, but the object is outside the arbiter's own in-path corridor so
    the arbiter itself sees no hazard.  It must still pass the full brake through
    IMMEDIATELY.  Before the fix: 0.250, 0.500, 0.750, 1.000 -- 200 ms, about 3 m at
    15 m/s, of an emergency stop thrown away.
    """
    arbiter = SafetyArbiter()
    off_to_the_side = _track(track_id=7, distance_m=11.0, center_x=60.0)
    plan = MotionPlan(0.0, 0.0, "aeb_ttc_0.50s")
    cmd = ControlCommand(0.0, 1.0, 0.0)
    for index in range(4):
        result = arbiter.arbitrate(
            plan,
            cmd,
            _context(15.0, tracks=[off_to_the_side], timestamp_s=index * DT),
        )
        assert result.command.brake == pytest.approx(1.0), (
            "frame %d: brake %.3f" % (index, result.command.brake)
        )


def test_brake_release_is_still_rate_limited():
    """Removing the APPLY-rate limit must not remove the RELEASE-rate floor.

    The release floor only ever holds the brake on for longer, so it cannot
    attenuate anything; it is the one shaper left on the braking channel.
    """
    limits = _limits(brake_release_rate_per_s=8.0)
    arbiter = SafetyArbiter(limits)
    arbiter.arbitrate(MotionPlan(0.0, 0.0, "x"), ControlCommand(0.0, 1.0, 0.0), _context(10.0))
    result = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"), ControlCommand(0.0, 0.0, 0.0), _context(10.0)
    )
    assert result.command.brake == pytest.approx(1.0 - limits.brake_release_rate_per_s * DT)
    assert "brake_release_limited" in arbiter.last_shaping


# --------------------------------------------------------------------------- #
# Command sanitation
# --------------------------------------------------------------------------- #


def test_nan_command_becomes_a_controlled_stop_not_full_braking():
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "ok"),
        ControlCommand(0.0, float("nan"), 0.0),
        _context(10.0),
    )
    assert result.state is SafetyState.MIN_RISK_MANEUVER
    assert "command_not_finite" in result.violations
    expected = arbiter.limits.mrm_decel_mps2 / arbiter.limits.brake_authority_mps2
    assert result.command.brake == pytest.approx(expected, abs=1e-6)
    assert result.command.brake < 1.0


def test_negative_throttle_is_rejected():
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "ok"), ControlCommand(-0.5, 0.0, 0.0), _context(10.0)
    )
    assert result.command.throttle == 0.0
    assert any("throttle_out_of_range" in v for v in result.violations)


def test_simultaneous_throttle_and_brake_is_a_violation():
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "ok"), ControlCommand(0.5, 0.5, 0.0), _context(10.0)
    )
    assert "command_throttle_brake_conflict" in result.violations
    assert result.command.throttle == 0.0


# --------------------------------------------------------------------------- #
# Lateral envelope -- ADAS-DEC-12 / ADAS-DEC-14
# --------------------------------------------------------------------------- #


def test_steering_slew_rate_is_enforced_on_the_physical_angle():
    limits = _limits(max_steering_rate_rad_s=0.5, max_road_wheel_rad=0.436)
    arbiter = SafetyArbiter(limits)
    result = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "ok"), ControlCommand(0.0, 0.0, 1.0), _context(0.0)
    )
    expected = (limits.max_steering_rate_rad_s * DT) / limits.max_road_wheel_rad
    assert result.command.steering == pytest.approx(expected, rel=1e-6)
    assert any("steering_rate" in v for v in result.violations)


def test_lateral_acceleration_cap_binds_at_speed():
    limits = _limits(max_lateral_accel_mps2=4.5, wheelbase_m=2.8, max_steering_rate_rad_s=10.0)
    arbiter = SafetyArbiter(limits)
    speed = 30.0
    result = arbiter.arbitrate(
        MotionPlan(speed, 0.0, "ok"), ControlCommand(0.0, 0.0, 1.0), _context(speed)
    )
    angle = abs(result.command.steering) * limits.max_road_wheel_rad
    achieved = (speed ** 2) * math.tan(angle) / limits.wheelbase_m
    assert achieved <= limits.max_lateral_accel_mps2 + 1e-6
    assert any("lateral_accel" in v for v in result.violations)


def test_steering_rate_bounded_over_an_adversarial_sequence():
    rng = random.Random(SEED + 1)
    limits = _limits(max_steering_rate_rad_s=0.5)
    arbiter = SafetyArbiter(limits)
    previous = 0.0
    for _ in range(2000):
        result = arbiter.arbitrate(
            MotionPlan(10.0, 0.0, "ok"),
            ControlCommand(0.0, 0.0, rng.choice([-1.0, 1.0, rng.uniform(-1, 1)])),
            _context(10.0),
        )
        angle = result.command.steering * limits.max_road_wheel_rad
        assert abs(angle - previous) <= limits.max_steering_rate_rad_s * DT + 1e-9
        previous = angle


def test_lane_departure_is_enforced_when_a_metric_offset_is_available():
    arbiter = SafetyArbiter(_limits(max_lateral_offset_m=1.5))
    inside = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "ok"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(10.0, lateral_offset_m=0.4),
    )
    assert arbiter.lane_offset_available is True
    assert not any("lane_departure" in v for v in inside.violations)

    outside = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "ok"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(10.0, lateral_offset_m=-2.1),
    )
    assert any("lane_departure" in v for v in outside.violations)
    assert outside.state is not SafetyState.NOMINAL
    assert outside.command.throttle == 0.0


def test_lane_departure_reports_unavailable_rather_than_passing_silently():
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "ok"), ControlCommand(0.2, 0.0, 0.0), _context(10.0)
    )
    assert arbiter.lane_offset_available is False
    assert "lane_offset_unavailable" in result.reason
    assert not any("lane_departure" in v for v in result.violations)


def test_measured_jerk_limit_is_enforced():
    """A speed trace with a large second difference must be flagged."""
    arbiter = SafetyArbiter(_limits(max_jerk_mps3=4.0))
    plan = MotionPlan(10.0, 0.0, "ok")
    cmd = ControlCommand(0.0, 0.0, 0.0)
    for speed in (10.0, 10.0, 10.1):  # 0 then +2 m/s^2 -> jerk 40 m/s^3
        result = arbiter.arbitrate(plan, cmd, _context(speed))
    assert any("jerk_" in v for v in result.violations)


# --------------------------------------------------------------------------- #
# Independence -- ADAS-DEC-13
# --------------------------------------------------------------------------- #


def test_arbiter_brakes_on_a_hazard_the_plan_denies():
    """Plan says cruise at full throttle; the raw tracks say 5 m and closing."""
    arbiter = SafetyArbiter()
    # Two frames so the arbiter's own filter can measure the closing rate.
    arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "cruise_clear"),
        ControlCommand(1.0, 0.0, 0.0),
        _context(15.0, tracks=[_track(distance_m=5.5, velocity_mps=0.0)]),
    )
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "cruise_clear"),
        ControlCommand(1.0, 0.0, 0.0),
        _context(15.0, tracks=[_track(distance_m=5.0, velocity_mps=0.0)]),
    )
    assert result.command.throttle == 0.0
    assert result.command.brake > 0.3
    assert result.state is SafetyState.MIN_RISK_MANEUVER


def test_arbiter_ignores_tracker_velocity_mps():
    """A slanderous tracker velocity must not by itself trigger braking.

    The arbiter now SEEDS a new track's rate rather than starting it at zero, so
    frame 0 no longer reports ``rate = 0``. The seed is ``-ego_speed`` -- the safe
    prior that an unknown object is stationary in the world -- which comes from the
    ego state and is bounded by it. It is never the tracker's number: 1230 m/s of
    claimed closing speed leaves no trace at all.
    """
    arbiter = SafetyArbiter()
    for index in range(25):
        result = arbiter.arbitrate(
            MotionPlan(10.0, 0.0, "cruise"),
            ControlCommand(0.2, 0.0, 0.0),
            _context(10.0, tracks=[_track(distance_m=60.0, velocity_mps=1230.0)]),
        )
        assert arbiter.last_lead is not None
        if index == 0:
            assert arbiter.last_lead.range_rate_mps == pytest.approx(-10.0)
            assert arbiter.last_lead.rate_updates == 1
        assert abs(arbiter.last_lead.range_rate_mps) <= 10.0 + 1e-9
        assert result.state is SafetyState.NOMINAL
    # A range that never changes converges the seeded rate back to zero.
    assert abs(arbiter.last_lead.range_rate_mps) < 0.5
    assert result.command.throttle == pytest.approx(0.2)


def test_arbiter_selects_by_time_to_collision_not_by_nearest():
    """A distant object closing fast outranks a near one keeping pace."""
    arbiter = SafetyArbiter()
    near_id, far_id = 1, 2
    gap_far = 60.0
    for _ in range(12):
        gap_far -= 20.0 * DT
        arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"),
            ControlCommand(0.2, 0.0, 0.0),
            _context(
                15.0,
                tracks=[
                    _track(track_id=near_id, distance_m=25.0),
                    _track(track_id=far_id, distance_m=gap_far),
                ],
            ),
        )
    assert arbiter.last_lead is not None
    assert arbiter.last_lead.track_id == far_id


def _disagreeing_context(nearer_m: float, confidence: float = 0.9, pinhole_m: float = 40.0):
    return _context(
        15.0,
        tracks=[_track(distance_m=pinhole_m)],
        independent_ranges={
            1: RangeEstimate(
                distance_m=nearer_m, confidence=confidence, source=RangeSource.DEPTH_MODEL
            )
        },
    )


def test_range_channel_disagreement_degrades_and_needs_corroboration():
    """A disagreement degrades at once but is only ADOPTED after K frames.

    A monocular depth model that under-reads range used to drive the AEB directly,
    with no plausibility test: one frame of ``depth = 15 m`` against a 40 m pinhole
    was enough. Now the first ``range_corroboration_frames - 1`` disagreeing frames
    report the disagreement and force LIMITED while still using the PINHOLE range,
    so a single-frame outlier cannot reach the emergency path.
    """
    limits = _limits(range_corroboration_frames=3)
    arbiter = SafetyArbiter(limits)
    for index in range(limits.range_corroboration_frames - 1):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"), ControlCommand(0.2, 0.0, 0.0), _disagreeing_context(15.0)
        )
        assert any("uncorroborated" in v for v in result.violations), index
        assert result.state is SafetyState.LIMITED
        assert arbiter.last_lead.distance_m == pytest.approx(40.0, abs=1.0)
        assert arbiter.last_lead.source is RangeSource.PINHOLE

    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"), ControlCommand(0.2, 0.0, 0.0), _disagreeing_context(15.0)
    )
    assert any(
        "range_channel_disagreement" in v and "uncorroborated" not in v for v in result.violations
    )
    assert arbiter.last_lead.distance_m == pytest.approx(15.0)
    assert arbiter.last_lead.source is RangeSource.FUSED
    assert result.state is not SafetyState.NOMINAL


def test_a_single_frame_depth_outlier_never_reaches_the_emergency_path():
    """One phantom close reading at highway speed must not command a full stop."""
    arbiter = SafetyArbiter()
    for _ in range(6):
        arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"),
            ControlCommand(0.2, 0.0, 0.0),
            _context(
                15.0,
                tracks=[_track(distance_m=40.0)],
                independent_ranges={
                    1: RangeEstimate(
                        distance_m=39.0, confidence=0.9, source=RangeSource.DEPTH_MODEL
                    )
                },
            ),
        )
    outlier = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"), ControlCommand(0.2, 0.0, 0.0), _disagreeing_context(3.0)
    )
    assert outlier.state is not SafetyState.MIN_RISK_MANEUVER
    assert outlier.command.brake == 0.0
    assert arbiter.last_lead.source is RangeSource.PINHOLE
    assert arbiter.last_lead.distance_m > 30.0, (
        "the 3 m outlier reached the hazard maths: %.2f m" % arbiter.last_lead.distance_m
    )


def test_switching_range_channel_does_not_manufacture_a_closing_rate():
    """The reviewer's /tmp/rev_more.py part B, as a permanent test.

    A stationary car 60 m ahead by pinhole, 40 m by depth, ego at 15 m/s: a 50%
    disagreement, which is the normal case for a monocular depth model. Adopting the
    nearer channel after corroboration is a discontinuity in the SIGNAL, not motion
    of the object -- differencing across it gave -45 m/s of "closing" for an ego
    doing 15 m/s and fired the AEB. The old code additionally counted the
    disagreement as a fault and latched DISENGAGE at frame 39.

    Required behaviour: LIMITED for the whole run, throttle cut, no emergency stop,
    never DISENGAGE.
    """
    arbiter = SafetyArbiter(_limits(disengage_after_frames=40))
    states = []
    for index in range(60):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "cruise"),
            ControlCommand(0.3, 0.0, 0.0),
            _context(
                15.0,
                tracks=[_track(track_id=1, distance_m=60.0)],
                timestamp_s=index * DT,
                independent_ranges={
                    1: RangeEstimate(
                        distance_m=40.0, confidence=0.8, source=RangeSource.DEPTH_MODEL
                    )
                },
            ),
        )
        states.append(result.state)
        assert arbiter.last_lead.range_rate_mps >= -15.0 - 1e-9, (
            "frame %d: manufactured rate %.1f m/s" % (index, arbiter.last_lead.range_rate_mps)
        )
    assert set(states) == {SafetyState.LIMITED}, sorted({s.value for s in states})
    assert result.command.throttle == 0.0
    assert result.command.brake == 0.0
    assert arbiter.fault_streak == 0


def test_a_low_confidence_second_channel_is_discarded_not_down_weighted():
    limits = _limits(min_range_confidence=0.35)
    arbiter = SafetyArbiter(limits)
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 0.0),
        _disagreeing_context(15.0, confidence=0.20),
    )
    assert any("range_channel_low_confidence" in v for v in result.violations)
    assert arbiter.last_lead.source is RangeSource.PINHOLE
    assert arbiter.last_lead.distance_m == pytest.approx(40.0, abs=1.0)


def test_a_farther_second_channel_is_never_adopted():
    """Raising the range on a disagreeing channel's word would reduce braking."""
    arbiter = SafetyArbiter(_limits(range_corroboration_frames=1))
    for _ in range(6):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"),
            ControlCommand(0.2, 0.0, 0.0),
            _disagreeing_context(90.0, pinhole_m=40.0),
        )
    assert any("farther" in v for v in result.violations)
    assert arbiter.last_lead.source is RangeSource.PINHOLE
    assert arbiter.last_lead.distance_m == pytest.approx(40.0, abs=1.0)


def test_an_absent_second_channel_is_neither_a_violation_nor_a_fused_source():
    """The depth channel may be switched off; that is not a finding of any kind."""
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(15.0, tracks=[_track(distance_m=40.0)], independent_ranges=None),
    )
    assert result.violations == []
    assert result.state is SafetyState.NOMINAL
    assert arbiter.last_lead.source is RangeSource.PINHOLE


def test_an_unavailable_range_estimate_is_treated_as_no_second_channel():
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(
            15.0,
            tracks=[_track(distance_m=40.0)],
            independent_ranges={
                1: RangeEstimate(distance_m=0.0, confidence=0.0, source=RangeSource.UNAVAILABLE)
            },
        ),
    )
    assert result.violations == []
    assert arbiter.last_lead.source is RangeSource.PINHOLE


def test_agreeing_range_channels_do_not_degrade():
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(
            15.0,
            tracks=[_track(distance_m=40.0)],
            independent_ranges={
                1: RangeEstimate(distance_m=38.0, confidence=0.7, source=RangeSource.DEPTH_MODEL)
            },
        ),
    )
    assert not any("range_channel_disagreement" in v for v in result.violations)
    assert result.state is SafetyState.NOMINAL


def test_range_teleport_uses_the_bounded_safe_prior_not_the_difference_quotient():
    """The 68 m -> 6.8 m association teleport used to imply 1230 m/s of closing.

    The re-init still fires and is still reported, and the rate the arbiter adopts
    is still nowhere near the raw difference quotient. What CHANGED is the value it
    re-initialises to: ``-ego_speed`` (the safe prior that the thing 6.8 m ahead is
    stationary) instead of 0.0 (the optimistic prior that it is keeping pace). A box
    measured 6.8 m ahead at 15 m/s is an emergency whichever way the association
    got there, so the arbiter brakes -- and its authority is still bounded by
    ``max_deceleration_mps2``, not by a 1230 m/s fantasy.
    """
    arbiter = SafetyArbiter()
    for _ in range(6):
        arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"),
            ControlCommand(0.2, 0.0, 0.0),
            _context(15.0, tracks=[_track(distance_m=68.3)]),
        )
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(15.0, tracks=[_track(distance_m=6.8)]),
    )
    assert any("range_jump" in v for v in result.violations)
    assert arbiter.last_lead.range_rate_mps == pytest.approx(-15.0)
    assert abs(arbiter.last_lead.range_rate_mps) <= 15.0 + 1e-9
    assert result.command.throttle == 0.0


def test_a_teleport_to_a_farther_range_does_not_brake():
    """The safe prior must not turn every re-init into a brake application."""
    arbiter = SafetyArbiter()
    for _ in range(6):
        arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"),
            ControlCommand(0.2, 0.0, 0.0),
            _context(15.0, tracks=[_track(distance_m=30.0)]),
        )
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(15.0, tracks=[_track(distance_m=95.0)]),
    )
    assert any("range_jump" in v for v in result.violations)
    assert arbiter.last_lead.range_rate_mps == pytest.approx(-15.0)
    assert result.command.brake == 0.0


def test_out_of_path_object_is_not_a_lead():
    arbiter = SafetyArbiter()
    edge = _track(track_id=5, distance_m=1.5, center_x=20.0)
    result = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(10.0, tracks=[edge], ego_lane_half_width_frac=0.2),
    )
    assert arbiter.last_lead is None
    assert result.state is SafetyState.NOMINAL


# --------------------------------------------------------------------------- #
# The arbiter's in-path corridor is its own geometry
# --------------------------------------------------------------------------- #


def _lane(center_px: float, confidence: float, is_mock: bool) -> LaneModel:
    return LaneModel(
        left_coeffs=(0.0, 0.0, 0.0),
        right_coeffs=(0.0, 0.0, 0.0),
        lane_center_px=center_px,
        curvature_m=0.0,
        confidence=confidence,
        is_mock=is_mock,
    )


def _closing_lead_scene(lane: Optional[LaneModel], frames: int = 12):
    """One lead dead ahead at image centre, closing at 15 m/s, planner cruising.

    ``in_ego_lane`` is deliberately False: perception did not label it. The planner
    is commanding throttle 0.6 throughout, so anything the arbiter does is the
    arbiter's own decision.
    """
    arbiter = SafetyArbiter()
    plan = MotionPlan(15.0, 0.0, "cruise_clear")
    distance_m = 30.0
    result = None
    for index in range(frames):
        result = arbiter.arbitrate(
            plan,
            ControlCommand(0.6, 0.0, 0.0),
            _context(
                15.0,
                tracks=[_track(track_id=1, distance_m=distance_m, in_ego_lane=False)],
                timestamp_s=index * DT,
                lane=lane,
                ego_lane_half_width_frac=0.20,
            ),
        )
        distance_m -= 15.0 * DT
    return arbiter, result


def test_a_wrong_lane_centre_cannot_hide_a_lead_from_the_arbiter():
    """The blocker repro (/tmp/rev_mocklane.py), as a permanent test.

    ``_in_path`` used to short-circuit on ``TrackedObject.in_ego_lane`` -- computed
    by the tracker from the same lane model the planner reads -- and otherwise test
    an image band centred on ``lane.lane_center_px``. A lane centre reported at
    180 px instead of 640 px (reachable from the real UFLD path, which synthesises a
    missing boundary at a fixed offset) therefore removed a real closing lead from
    the arbiter's view entirely, and it passed the planner's throttle 0.6 straight
    through in NOMINAL. An independent backstop that inherits the perception error it
    exists to catch is not independent.

    The corridor is now anchored on the IMAGE CENTRE. A lane model can only add a
    second anchor -- widening -- and only when it is real and confident.
    """
    good = _lane(640.0, confidence=0.9, is_mock=False)
    bad = _lane(180.0, confidence=0.9, is_mock=False)
    mock = _lane(180.0, confidence=0.0, is_mock=True)

    outcomes = {}
    for name, lane in (("good", good), ("bad", bad), ("mock", mock), ("none", None)):
        arbiter, result = _closing_lead_scene(lane)
        assert arbiter.last_lead is not None, "%s lane hid the lead" % name
        assert result.command.throttle == 0.0, "%s lane let the throttle through" % name
        assert result.command.brake > 0.5, "%s lane: brake %.2f" % (name, result.command.brake)
        assert result.state is SafetyState.MIN_RISK_MANEUVER, name
        outcomes[name] = round(result.command.brake, 6)

    assert len(set(outcomes.values())) == 1, (
        "the lane model changed the arbiter's braking decision: %r" % outcomes
    )


def test_the_arbiter_does_not_trust_the_trackers_in_ego_lane_flag():
    """The flag is common-mode with the planner, so the arbiter ignores it.

    It is not consulted in EITHER direction: it cannot add an object the arbiter's
    own corridor excludes, and (see the previous test) its absence cannot remove one
    the corridor includes.
    """
    arbiter = SafetyArbiter()
    far_left = _track(track_id=3, distance_m=6.0, center_x=40.0, in_ego_lane=True)
    result = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"), ControlCommand(0.3, 0.0, 0.0), _context(10.0, tracks=[far_left])
    )
    assert arbiter.last_lead is None
    assert result.state is SafetyState.NOMINAL


def test_the_arbiter_corridor_is_never_narrower_than_the_planners():
    """``min_in_path_half_width_frac`` is a floor, and the context value can widen."""
    arbiter = SafetyArbiter(_limits(min_in_path_half_width_frac=0.30))
    # 0.30 of 1280 = 384 px each side of centre: the corridor is [256, 1024].
    just_inside = _track(track_id=1, distance_m=8.0, center_x=1010.0)
    result = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"),
        ControlCommand(0.3, 0.0, 0.0),
        # The planner would have used a 0.05 band; the arbiter must not.
        _context(10.0, tracks=[just_inside], ego_lane_half_width_frac=0.05),
    )
    assert arbiter.last_lead is not None
    assert result.command.throttle == 0.0

    wider = SafetyArbiter(_limits(min_in_path_half_width_frac=0.30))
    outside_both = _track(track_id=1, distance_m=8.0, center_x=1250.0)
    wider.arbitrate(
        MotionPlan(10.0, 0.0, "x"),
        ControlCommand(0.3, 0.0, 0.0),
        _context(10.0, tracks=[outside_both], ego_lane_half_width_frac=0.05),
    )
    assert wider.last_lead is None
    # ... until the caller asks for a band wide enough to contain it.
    widest = SafetyArbiter(_limits(min_in_path_half_width_frac=0.30))
    widest.arbitrate(
        MotionPlan(10.0, 0.0, "x"),
        ControlCommand(0.3, 0.0, 0.0),
        _context(10.0, tracks=[outside_both], ego_lane_half_width_frac=0.49),
    )
    assert widest.last_lead is not None


def test_a_box_that_only_overlaps_the_corridor_edge_is_in_path():
    """Overlap, not centre: better to consider an out-of-lane object than miss one."""
    arbiter = SafetyArbiter(_limits(min_in_path_half_width_frac=0.30))
    # Corridor right edge is 1024 px. Centre at 1080 is outside it; the box spans
    # 1020..1140, so part of the vehicle is in the ego's path.
    straddling = _track(track_id=1, distance_m=8.0, center_x=1080.0)
    arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"), ControlCommand(0.3, 0.0, 0.0), _context(10.0, tracks=[straddling])
    )
    assert arbiter.last_lead is not None


def test_a_confident_lane_widens_the_corridor_and_never_moves_it():
    """A trusted lane centre adds objects; it can never remove them."""
    off_centre = _track(track_id=1, distance_m=8.0, center_x=1150.0)
    without = SafetyArbiter()
    without.arbitrate(
        MotionPlan(10.0, 0.0, "x"), ControlCommand(0.3, 0.0, 0.0), _context(10.0, tracks=[off_centre])
    )
    assert without.last_lead is None

    with_lane = SafetyArbiter()
    with_lane.arbitrate(
        MotionPlan(10.0, 0.0, "x"),
        ControlCommand(0.3, 0.0, 0.0),
        _context(10.0, tracks=[off_centre], lane=_lane(1100.0, confidence=0.9, is_mock=False)),
    )
    assert with_lane.last_lead is not None

    # A mock or unconfident lane at the same place adds nothing.
    for lane in (_lane(1100.0, 0.9, True), _lane(1100.0, 0.1, False)):
        untrusted = SafetyArbiter()
        untrusted.arbitrate(
            MotionPlan(10.0, 0.0, "x"),
            ControlCommand(0.3, 0.0, 0.0),
            _context(10.0, tracks=[off_centre], lane=lane),
        )
        assert untrusted.last_lead is None


# --------------------------------------------------------------------------- #
# Ego state and timing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "ego",
    [None, EgoState(speed_mps=10.0, valid=False), EgoState(speed_mps=float("nan"), valid=True),
     EgoState(speed_mps=-3.0, valid=True)],
)
def test_unusable_ego_state_forces_a_minimum_risk_manoeuvre(ego):
    arbiter = SafetyArbiter()
    context = SafetyContext(ego=ego, dt_s=DT, frame_width_px=WIDTH)
    result = arbiter.arbitrate(MotionPlan(10.0, 0.0, "x"), ControlCommand(1.0, 0.0, 0.0), context)
    assert result.state is SafetyState.MIN_RISK_MANEUVER
    assert result.command.throttle == 0.0
    assert result.command.brake > 0.0


def test_stale_timestamp_is_flagged():
    arbiter = SafetyArbiter()
    base = SafetyContext(ego=_ego(10.0), dt_s=DT, timestamp_s=100.0, frame_width_px=WIDTH)
    arbiter.arbitrate(MotionPlan(10.0, 0.0, "x"), ControlCommand(0.1, 0.0, 0.0), base)
    late = SafetyContext(ego=_ego(10.0), dt_s=DT, timestamp_s=101.0, frame_width_px=WIDTH)
    result = arbiter.arbitrate(MotionPlan(10.0, 0.0, "x"), ControlCommand(0.1, 0.0, 0.0), late)
    assert any("timing_stale_input" in v for v in result.violations)


def test_non_monotonic_timestamp_is_flagged():
    arbiter = SafetyArbiter()
    base = SafetyContext(ego=_ego(10.0), dt_s=DT, timestamp_s=100.0, frame_width_px=WIDTH)
    arbiter.arbitrate(MotionPlan(10.0, 0.0, "x"), ControlCommand(0.1, 0.0, 0.0), base)
    back = SafetyContext(ego=_ego(10.0), dt_s=DT, timestamp_s=99.9, frame_width_px=WIDTH)
    result = arbiter.arbitrate(MotionPlan(10.0, 0.0, "x"), ControlCommand(0.1, 0.0, 0.0), back)
    assert "timing_non_monotonic_timestamp" in result.violations


def test_measured_timestamps_are_preferred_over_the_nominal_dt():
    """A dropped frame must widen the rate-limit window, not be silently ignored."""
    limits = _limits(max_steering_rate_rad_s=0.5)
    arbiter = SafetyArbiter(limits)
    context_a = SafetyContext(ego=_ego(10.0), dt_s=0.05, timestamp_s=10.0, frame_width_px=WIDTH)
    arbiter.arbitrate(MotionPlan(10.0, 0.0, "x"), ControlCommand(0.0, 0.0, 0.0), context_a)
    context_b = SafetyContext(ego=_ego(10.0), dt_s=0.05, timestamp_s=10.2, frame_width_px=WIDTH)
    result = arbiter.arbitrate(MotionPlan(10.0, 0.0, "x"), ControlCommand(0.0, 0.0, 1.0), context_b)
    allowed = (limits.max_steering_rate_rad_s * 0.2) / limits.max_road_wheel_rad
    assert result.command.steering == pytest.approx(allowed, rel=1e-6)


def test_measured_acceleration_limit_is_enforced():
    arbiter = SafetyArbiter(_limits(max_acceleration_mps2=3.0))
    arbiter.arbitrate(MotionPlan(10.0, 0.0, "x"), ControlCommand(0.1, 0.0, 0.0), _context(10.0))
    result = arbiter.arbitrate(
        MotionPlan(11.0, 0.0, "x"), ControlCommand(0.1, 0.0, 0.0), _context(11.0)
    )
    assert any("measured_accel" in v for v in result.violations)


def test_unjustified_hard_deceleration_is_flagged_but_a_hazard_stop_is_not():
    arbiter = SafetyArbiter(_limits(max_deceleration_mps2=8.0))
    arbiter.arbitrate(MotionPlan(10.0, 0.0, "x"), ControlCommand(0.0, 0.1, 0.0), _context(20.0))
    unjustified = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"), ControlCommand(0.0, 0.1, 0.0), _context(19.0)
    )
    assert any("measured_decel" in v and "unjustified" in v for v in unjustified.violations)

    hazard_arbiter = SafetyArbiter(_limits(max_deceleration_mps2=8.0))
    hazard_arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"),
        ControlCommand(0.0, 0.5, 0.0),
        _context(20.0, tracks=[_track(distance_m=8.0)]),
    )
    justified = hazard_arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"),
        ControlCommand(0.0, 0.5, 0.0),
        _context(19.0, tracks=[_track(distance_m=7.0)]),
    )
    assert not any("unjustified" in v for v in justified.violations)


# --------------------------------------------------------------------------- #
# Degradation state machine -- ADAS-DEC-04
# --------------------------------------------------------------------------- #


def test_one_perception_dropout_enters_limited():
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.8, 0.0, 0.0),
        _context(15.0, perception_ok=False),
    )
    assert result.state is SafetyState.LIMITED
    assert result.command.throttle == 0.0
    assert result.command.brake > 0.0


def test_three_perception_dropouts_enter_a_minimum_risk_manoeuvre():
    arbiter = SafetyArbiter()
    states = []
    for _ in range(3):
        states.append(
            arbiter.arbitrate(
                MotionPlan(15.0, 0.0, "x"),
                ControlCommand(0.8, 0.0, 0.0),
                _context(15.0, perception_ok=False),
            ).state
        )
    assert states[0] is SafetyState.LIMITED
    assert states[-1] is SafetyState.MIN_RISK_MANEUVER


def test_an_empty_track_list_with_healthy_perception_is_not_a_dropout():
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "cruise"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(15.0, tracks=[], perception_ok=True),
    )
    assert result.state is SafetyState.NOMINAL
    assert result.command.throttle == pytest.approx(0.2)


def test_recovery_requires_a_run_of_clean_frames():
    limits = _limits(recovery_frames=10)
    arbiter = SafetyArbiter(limits)
    arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.5, 0.0, 0.0),
        _context(15.0, perception_ok=False),
    )
    assert arbiter.state is SafetyState.LIMITED
    for index in range(1, limits.recovery_frames):
        arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"), ControlCommand(0.0, 0.0, 0.0), _context(15.0)
        )
        assert arbiter.state is SafetyState.LIMITED, "recovered after only %d frames" % index
    final = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"), ControlCommand(0.0, 0.0, 0.0), _context(15.0)
    )
    assert final.state is SafetyState.NOMINAL


def test_disengage_latches_until_reset():
    limits = _limits(disengage_after_frames=5)
    arbiter = SafetyArbiter(limits)
    for _ in range(6):
        arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"),
            ControlCommand(0.5, 0.0, 0.0),
            _context(15.0, perception_ok=False),
        )
    assert arbiter.state is SafetyState.DISENGAGE
    for _ in range(100):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"), ControlCommand(0.5, 0.0, 0.0), _context(15.0)
        )
        assert result.state is SafetyState.DISENGAGE
        assert result.command.throttle == 0.0
    arbiter.reset()
    assert arbiter.state is SafetyState.NOMINAL


# --------------------------------------------------------------------------- #
# Only HEALTH faults may latch DISENGAGE
# --------------------------------------------------------------------------- #


def test_a_fast_frame_is_not_a_finding_at_all():
    """The blocker repro (/tmp/rev_fps0.py), reduced to its cause.

    An unpaced run (``--fps 0``) produces 2 ms frames. ``_resolve_dt`` reported
    ``timing_dt_out_of_range`` for every one of them because it was BELOW
    ``min_dt_s``, and 40 in a row latched DISENGAGE at frame 47 of a 120-frame run
    on an empty road with a valid ego state. Running well is not a fault: the dt is
    clamped up, silently.
    """
    arbiter = SafetyArbiter()
    for index in range(200):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "cruise_clear"),
            ControlCommand(0.0, 0.0, 0.0),
            _context(15.0, dt_s=0.002, timestamp_s=index * 0.002),
        )
        assert result.violations == [], "frame %d: %r" % (index, result.violations)
        assert result.state is SafetyState.NOMINAL, "frame %d: %s" % (index, result.state)
    assert arbiter.fault_streak == 0


def test_a_dt_above_the_maximum_is_still_a_health_fault():
    """Clamping the fast end must not stop reporting the slow end."""
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"), ControlCommand(0.0, 0.0, 0.0), _context(15.0, dt_s=1.5)
    )
    assert any("timing_dt_above_max" in v for v in result.violations)
    assert arbiter.fault_streak == 1


@pytest.mark.parametrize(
    "name, kwargs, command, plan",
    [
        (
            "lateral_accel",
            dict(ego_speed_mps=15.0),
            ControlCommand(0.2, 0.0, 6.0 / 25.0),
            MotionPlan(15.0, 6.0, "cruise"),
        ),
        (
            "range_channel_disagreement",
            dict(
                ego_speed_mps=5.0,
                tracks=[_track(distance_m=60.0)],
                independent_ranges={
                    1: RangeEstimate(
                        distance_m=40.0, confidence=0.9, source=RangeSource.DEPTH_MODEL
                    )
                },
            ),
            ControlCommand(0.2, 0.0, 0.0),
            MotionPlan(5.0, 0.0, "cruise"),
        ),
        (
            "plan_over_speed",
            dict(ego_speed_mps=30.0),
            ControlCommand(0.2, 0.0, 0.0),
            MotionPlan(40.0, 0.0, "cruise"),
        ),
        (
            "camera_uncalibrated",
            dict(ego_speed_mps=15.0, camera_calibrated=False),
            ControlCommand(0.2, 0.0, 0.0),
            MotionPlan(15.0, 0.0, "cruise"),
        ),
    ],
)
def test_a_mitigated_condition_never_latches_disengage(name, kwargs, command, plan):
    """The blocker: ordinary, successfully-mitigated clamps used to disengage.

    ``_decide_state`` did ``if faults: self._violation_streak += 1`` where ``faults``
    included clamps the arbiter had already applied and handled. 40 frames -- two
    seconds at 20 Hz -- of any of these latched the terminal state, from which
    nothing in the runner ever recovers. Each of these conditions must degrade the
    state and be logged, and must never reach the latch.
    """
    limits = _limits(disengage_after_frames=5, range_corroboration_frames=1)
    arbiter = SafetyArbiter(limits)
    result = None
    for index in range(60):
        result = arbiter.arbitrate(plan, command, _context(timestamp_s=index * DT, **kwargs))
    assert arbiter.state is not SafetyState.DISENGAGE, name
    assert arbiter.state is SafetyState.LIMITED, name
    assert arbiter.fault_streak == 0, "%s reached the health-fault counter" % name
    assert result.violations, "%s stopped being reported at all" % name


def test_a_sustained_health_fault_still_latches_disengage():
    """Splitting the counter must not disarm it."""
    limits = _limits(disengage_after_frames=5)
    arbiter = SafetyArbiter(limits)
    for _ in range(limits.disengage_after_frames):
        arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"),
            ControlCommand(0.5, 0.0, 0.0),
            _context(15.0, perception_ok=False),
        )
    assert arbiter.state is SafetyState.DISENGAGE
    assert arbiter.fault_streak >= limits.disengage_after_frames


def test_a_missing_plan_is_a_health_fault():
    limits = _limits(disengage_after_frames=4)
    arbiter = SafetyArbiter(limits)
    for _ in range(4):
        result = arbiter.arbitrate(None, ControlCommand(0.5, 0.0, 0.0), _context(15.0))
    assert "plan_missing" in result.violations
    assert result.state is SafetyState.DISENGAGE


def test_an_uncalibrated_camera_is_reported_and_cannot_be_nominal():
    """``CameraConfig(calibrated=False)`` is the shipped default.

    Every metric range the hazard maths uses is then an assumption. The arbiter says
    so and refuses to report NOMINAL while acting on it -- unless the deployment has
    opted in explicitly, the same shape as the ``--allow-mock`` gate.
    """
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "cruise"),
        ControlCommand(0.4, 0.0, 0.0),
        _context(15.0, camera_calibrated=False),
    )
    assert "camera_uncalibrated" in result.violations
    assert result.state is SafetyState.LIMITED
    assert result.command.throttle == 0.0

    allowed = SafetyArbiter(_limits(allow_uncalibrated_range=True))
    opted_in = allowed.arbitrate(
        MotionPlan(15.0, 0.0, "cruise"),
        ControlCommand(0.4, 0.0, 0.0),
        _context(15.0, camera_calibrated=False),
    )
    assert opted_in.state is SafetyState.NOMINAL
    assert "camera_uncalibrated" not in opted_in.violations

    silent = SafetyArbiter()
    unreported = silent.arbitrate(
        MotionPlan(15.0, 0.0, "cruise"), ControlCommand(0.4, 0.0, 0.0), _context(15.0)
    )
    assert unreported.state is SafetyState.NOMINAL
    assert "camera_uncalibrated" not in unreported.violations


# --------------------------------------------------------------------------- #
# The closing-rate seed -- the AEB backstop on frame 0 of a track
# --------------------------------------------------------------------------- #


def test_a_stationary_obstacle_brakes_on_the_first_frame_it_is_seen():
    """The repro (/tmp/rev_cutin.py frame 0), as a permanent test.

    A stationary car 15 m ahead, ego at 15 m/s, no track history: the arbiter used
    to report ``state=nominal ttc=inf rate=+0.00 rss=13.7`` and command brake 0.000,
    because a new track's rate started at zero -- the OPTIMISTIC prior -- which also
    made ``lead_speed = ego_speed + 0`` and collapsed the RSS minimum gap from
    ~27.8 m to 13.7 m so the headway test could not fire either. It was blind
    exactly when a hazard appeared, on every new track, every occlusion exit and
    every track-id change.
    """
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "cruise_clear"),
        ControlCommand(0.8, 0.0, 0.0),
        _context(15.0, tracks=[_track(track_id=1, distance_m=15.0)]),
    )
    lead = arbiter.last_lead
    assert lead is not None
    assert lead.range_rate_mps == pytest.approx(-15.0)
    assert lead.ttc_s < 1.0
    assert lead.rss_min_gap_m > 25.0
    assert result.state is SafetyState.MIN_RISK_MANEUVER
    assert result.command.throttle == 0.0
    assert result.command.brake > 0.5


def test_the_seed_comes_from_the_ego_state_and_is_bounded_by_it():
    filt = ArbiterLimits()  # noqa: F841 - documents that no limit tunes the seed
    arbiter = SafetyArbiter()
    for speed in (0.0, 5.0, 20.0):
        fresh = SafetyArbiter()
        fresh.arbitrate(
            MotionPlan(speed, 0.0, "x"),
            ControlCommand(0.0, 0.0, 0.0),
            _context(speed, tracks=[_track(track_id=1, distance_m=70.0)]),
        )
        assert fresh.last_lead.range_rate_mps == pytest.approx(-speed)
    assert arbiter.state is SafetyState.NOMINAL


def test_a_lead_that_keeps_pace_converges_off_the_seed_and_returns_to_nominal():
    """The safe prior must be transient, not a permanent phantom brake."""
    arbiter = SafetyArbiter()
    result = None
    for index in range(40):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "follow"),
            ControlCommand(0.2, 0.0, 0.0),
            _context(15.0, tracks=[_track(track_id=1, distance_m=45.0)], timestamp_s=index * DT),
        )
    assert abs(arbiter.last_lead.range_rate_mps) < 0.2
    assert result.state is SafetyState.NOMINAL
    assert result.command.brake == 0.0


# --------------------------------------------------------------------------- #
# Minimum-risk manoeuvre lateral behaviour
# --------------------------------------------------------------------------- #


def test_an_mrm_holds_the_steering_it_last_commanded_instead_of_straightening():
    """The repro (/tmp/rev_curve.py frame 59), as a permanent test.

    In MRM/DISENGAGE the arbiter replaced the steering with ``_last_good_steering``,
    which was only ever written in NOMINAL -- and NOMINAL requires zero findings, so
    in any curve where a lateral clamp is active it never updated and stayed at its
    reset value of 0.0. The arbiter commanded ``out_steer=0.000`` into a corner while
    braking at 3.5 m/s^2, then reported ``steering_rate_1.12_above_0.50`` against its
    own hold, which sustained the fault and fed the DISENGAGE counter.
    """
    limits = _limits(max_steering_rate_rad_s=10.0, mrm_straighten_speed_mps=1.0)
    arbiter = SafetyArbiter(limits)
    turning = ControlCommand(0.2, 0.0, 0.25)
    for index in range(10):
        nominal = arbiter.arbitrate(
            MotionPlan(8.0, 6.0, "lane_center"), turning, _context(8.0, timestamp_s=index * DT)
        )
    held = nominal.command.steering
    assert held > 0.05, "the setup never actually commanded a steering angle"

    # Perception drops out: MRM. The steering must stay on the path.
    for index in range(3):
        mrm = arbiter.arbitrate(
            MotionPlan(8.0, 6.0, "lane_center"),
            turning,
            _context(8.0, perception_ok=False, timestamp_s=(10 + index) * DT),
        )
    assert mrm.state is SafetyState.MIN_RISK_MANEUVER
    assert mrm.command.brake > 0.0
    assert mrm.command.steering == pytest.approx(held, abs=1e-6), (
        "MRM straightened the wheel: %.3f -> %.3f" % (held, mrm.command.steering)
    )


def test_an_mrm_entered_mid_corner_does_not_blame_itself_for_its_own_hold():
    """The self-sustaining part of the same defect.

    ``steering_rate`` is now raised against the previous REQUEST, so the difference
    between the arbiter's held output and the planner's steady request is no longer
    reported as a violation every frame.
    """
    arbiter = SafetyArbiter()
    steady = ControlCommand(0.2, 0.0, 6.0 / 25.0)
    plan = MotionPlan(15.0, 6.0, "lane_center")
    result = None
    for index in range(80):
        result = arbiter.arbitrate(plan, steady, _context(15.0, timestamp_s=index * DT))
    assert arbiter.state is SafetyState.LIMITED
    assert arbiter.state is not SafetyState.DISENGAGE
    assert not any("steering_rate" in v for v in result.violations), result.violations
    assert result.violations == ["lateral_accel_8.4_above_4.5"]
    # ... and the clamp is a real clamp: the output is at the lateral ceiling.
    ceiling = math.atan(
        arbiter.limits.max_lateral_accel_mps2 * arbiter.limits.wheelbase_m / (15.0 ** 2)
    )
    assert result.command.steering == pytest.approx(
        ceiling / arbiter.limits.max_road_wheel_rad, rel=1e-6
    )


def test_a_held_mrm_steering_is_re_checked_against_the_lateral_ceiling():
    """The hold goes back through the ceiling, it does not bypass it."""
    limits = _limits(max_steering_rate_rad_s=10.0)
    arbiter = SafetyArbiter(limits)
    for index in range(6):
        arbiter.arbitrate(
            MotionPlan(4.0, 0.0, "x"),
            ControlCommand(0.2, 0.0, 1.0),
            _context(4.0, timestamp_s=index * DT),
        )
    slow_hold = arbiter.arbitrate(
        MotionPlan(4.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 1.0),
        _context(4.0, perception_ok=False, timestamp_s=6 * DT),
    )
    fast = arbiter.arbitrate(
        MotionPlan(25.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 1.0),
        _context(25.0, perception_ok=False, timestamp_s=7 * DT),
    )
    angle = abs(fast.command.steering) * limits.max_road_wheel_rad
    achieved = (25.0 ** 2) * math.tan(angle) / limits.wheelbase_m
    assert achieved <= limits.max_lateral_accel_mps2 + 1e-6
    assert abs(fast.command.steering) < abs(slow_hold.command.steering)


def test_an_mrm_straightens_only_once_the_vehicle_has_essentially_stopped():
    limits = _limits(max_steering_rate_rad_s=10.0, mrm_straighten_speed_mps=1.0)
    arbiter = SafetyArbiter(limits)
    for index in range(6):
        turning = arbiter.arbitrate(
            MotionPlan(5.0, 0.0, "x"),
            ControlCommand(0.2, 0.0, 0.2),
            _context(5.0, timestamp_s=index * DT),
        )
    assert turning.command.steering == pytest.approx(0.2)

    # Perception drops out and the MRM brings the vehicle to a stop. The hold
    # persists while the vehicle is still moving...
    states = []
    for offset, speed in enumerate((5.0, 4.0, 2.0, 0.5)):
        stopping = arbiter.arbitrate(
            MotionPlan(0.0, 0.0, "x"),
            ControlCommand(0.0, 0.0, 0.2),
            _context(speed, perception_ok=False, timestamp_s=(6 + offset) * DT),
        )
        states.append((speed, stopping.state, stopping.command.steering))
    assert states[2][1] is SafetyState.MIN_RISK_MANEUVER
    assert states[2][2] == pytest.approx(0.2), states
    # ... and releases once it has essentially stopped.
    assert states[3][1] is SafetyState.MIN_RISK_MANEUVER
    assert states[3][2] == pytest.approx(0.0), states


def test_degraded_states_never_allow_throttle():
    rng = random.Random(SEED + 2)
    arbiter = SafetyArbiter()
    for _ in range(2000):
        result = arbiter.arbitrate(
            MotionPlan(rng.uniform(0.0, 30.0), 0.0, "x"),
            ControlCommand(rng.uniform(0.0, 1.0), 0.0, 0.0),
            _context(
                rng.uniform(0.0, 25.0),
                tracks=[_track(distance_m=rng.uniform(1.0, 80.0))] if rng.random() < 0.5 else [],
                perception_ok=rng.random() > 0.2,
            ),
        )
        if result.state is not SafetyState.NOMINAL:
            assert result.command.throttle == 0.0


# --------------------------------------------------------------------------- #
# Closed-loop scenarios
# --------------------------------------------------------------------------- #


@dataclass
class PointMassPlant:
    """Jerk-limited longitudinal plant.

    ``throttle = 1`` gives ``accel_authority_mps2`` and ``brake = 1`` gives
    ``-brake_authority_mps2``; the achieved acceleration slews toward that at
    ``max_jerk_mps3``, which is what a real pedal/booster lag looks like to first
    order. Speed integrates the achieved acceleration and is floored at zero.

    Units: m, m/s, m/s^2, m/s^3, s.
    """

    speed_mps: float = 0.0
    accel_mps2: float = 0.0
    position_m: float = 0.0
    accel_authority_mps2: float = 2.5
    brake_authority_mps2: float = 8.0
    max_jerk_mps3: float = 3.5

    def step(self, command: ControlCommand, dt_s: float) -> None:
        target = (
            command.throttle * self.accel_authority_mps2
            - command.brake * self.brake_authority_mps2
        )
        step = self.max_jerk_mps3 * dt_s
        if target > self.accel_mps2 + step:
            self.accel_mps2 += step
        elif target < self.accel_mps2 - step:
            self.accel_mps2 -= step
        else:
            self.accel_mps2 = target
        new_speed = self.speed_mps + self.accel_mps2 * dt_s
        if new_speed <= 0.0:
            new_speed = 0.0
            self.accel_mps2 = 0.0
        self.speed_mps = new_speed
        self.position_m += self.speed_mps * dt_s


@dataclass
class FrameRecord:
    index: int
    ego_speed_mps: float
    gap_m: float
    command: ControlCommand
    state: SafetyState
    violations: List[str]
    plan_target_mps: float


def run_scenario(
    frames: int,
    ego_speed0: float,
    lead_gap0: Optional[float] = None,
    lead_speed_fn: Optional[Callable[[int, float], float]] = None,
    track_present_fn: Optional[Callable[[int], bool]] = None,
    perception_ok_fn: Optional[Callable[[int], bool]] = None,
    range_noise_m: float = 0.0,
    seed: int = SEED,
    dt_s: float = DT,
    arbiter_limits: Optional[ArbiterLimits] = None,
) -> List[FrameRecord]:
    """Run planner -> controller -> arbiter -> plant for ``frames`` steps.

    The lead vehicle, when present, starts ``lead_gap0`` metres ahead and moves at
    ``lead_speed_fn(index, t)`` m/s. The range handed to the perception layer is the
    true gap plus optional zero-mean noise, which is exactly what a box-height
    pinhole estimator produces.
    """
    rng = random.Random(seed)
    planner = BehaviorPlanner(cruise_speed_mps=15.0)
    controller = PIDLikeLongitudinalController(kp_speed=0.6, ki_speed=0.15)
    monitor = SafetyMonitor()
    if arbiter_limits is not None:
        monitor.arbiter = SafetyArbiter(arbiter_limits)

    plant = PointMassPlant(speed_mps=ego_speed0)
    lead_position_m = None if lead_gap0 is None else lead_gap0
    history: List[FrameRecord] = []

    for index in range(frames):
        time_s = index * dt_s
        perception_ok = perception_ok_fn(index) if perception_ok_fn else True
        present = track_present_fn(index) if track_present_fn else lead_gap0 is not None

        gap_m = float("inf")
        tracks: List[TrackedObject] = []
        if lead_position_m is not None:
            gap_m = lead_position_m - plant.position_m
            if present and perception_ok:
                measured = gap_m + (rng.gauss(0.0, range_noise_m) if range_noise_m else 0.0)
                tracks = [_track(track_id=1, distance_m=max(0.0, measured))]

        ego = EgoState(speed_mps=plant.speed_mps, valid=True, timestamp_s=time_s)
        plan = planner.plan(
            WIDTH,
            WIDTH / 2.0,
            tracks,
            ego=ego,
            perception_valid=perception_ok,
            dt_s=dt_s,
        )
        command = controller.to_command(plan, plant.speed_mps, dt_s=dt_s)
        context = SafetyContext(
            ego=ego,
            tracks=tracks,
            perception=PerceptionStatus(ok=perception_ok),
            dt_s=dt_s,
            timestamp_s=time_s,
            frame_width_px=WIDTH,
            frame_height_px=HEIGHT,
        )
        result = monitor.arbitrate(plan, command, context)

        history.append(
            FrameRecord(
                index=index,
                ego_speed_mps=plant.speed_mps,
                gap_m=gap_m,
                command=result.command,
                state=result.state,
                violations=list(result.violations),
                plan_target_mps=plan.target_speed_mps,
            )
        )

        plant.step(result.command, dt_s)
        if lead_position_m is not None:
            lead_speed = lead_speed_fn(index, time_s) if lead_speed_fn else 0.0
            lead_position_m += max(0.0, lead_speed) * dt_s

    return history


def _assert_no_collision(history: List[FrameRecord], floor_m: float = 0.0) -> None:
    worst = min(record.gap_m for record in history)
    assert worst > floor_m, "simulated collision: minimum gap %.2f m" % worst


def _assert_command_envelope(history: List[FrameRecord]) -> None:
    for record in history:
        assert 0.0 <= record.command.throttle <= 1.0
        assert 0.0 <= record.command.brake <= 1.0
        assert -1.0 <= record.command.steering <= 1.0
        assert not (record.command.throttle > 0.0 and record.command.brake > 0.0)


def test_scenario_stopped_lead_at_range():
    """Approach a stationary vehicle 80 m ahead at 15 m/s and stop behind it."""
    history = run_scenario(frames=1200, ego_speed0=15.0, lead_gap0=80.0)
    _assert_no_collision(history, floor_m=0.5)
    _assert_command_envelope(history)
    assert history[-1].ego_speed_mps < 0.5, "did not come to a stop"


def test_scenario_lead_brakes_hard():
    """Steady following, then the lead brakes at 6 m/s^2 to a standstill."""

    def lead_speed(index: int, time_s: float) -> float:
        if time_s < 6.0:
            return 12.0
        return max(0.0, 12.0 - 6.0 * (time_s - 6.0))

    history = run_scenario(
        frames=1400, ego_speed0=12.0, lead_gap0=36.0, lead_speed_fn=lead_speed
    )
    _assert_no_collision(history, floor_m=0.0)
    _assert_command_envelope(history)
    assert max(record.command.brake for record in history) > 0.3
    assert history[-1].ego_speed_mps < 0.5


def test_scenario_cut_in():
    """A slower vehicle enters the ego lane 20 m ahead with no track history.

    Before frame 40 the vehicle exists on the road but is not in the ego lane, so
    perception reports nothing; at frame 40 it appears at 20 m closing at 5 m/s and
    the arbiter has zero range history to work with.
    """

    def present(index: int) -> bool:
        return index >= 40

    history = run_scenario(
        frames=1200,
        ego_speed0=15.0,
        lead_gap0=30.0,
        lead_speed_fn=lambda i, t: 10.0,
        track_present_fn=present,
    )
    _assert_no_collision(history, floor_m=0.0)
    _assert_command_envelope(history)
    assert history[39].gap_m == pytest.approx(20.0, abs=1.0)
    assert history[40].command.throttle == 0.0, "throttle was not cut on the cut-in frame"
    # The arbiter has no range history on the cut-in frame, so its own alpha-beta
    # filter needs a few frames to see the closing rate. It must degrade well
    # inside a second.
    assert any(record.state is not SafetyState.NOMINAL for record in history[40:60])
    assert all(record.command.throttle == 0.0 for record in history[40:60])


def test_scenario_lead_disappears_does_not_step_the_throttle():
    """Losing a track must not produce a throttle or target-speed step."""

    def present(index: int) -> bool:
        return index < 200

    history = run_scenario(
        frames=600, ego_speed0=8.0, lead_gap0=25.0, lead_speed_fn=lambda i, t: 8.0,
        track_present_fn=present,
    )
    _assert_command_envelope(history)
    for previous, current in zip(history, history[1:]):
        assert current.plan_target_mps - previous.plan_target_mps <= 2.0 * DT + 1e-6
        assert current.command.throttle - previous.command.throttle <= (
            PIDLikeLongitudinalController().throttle_rate_per_s * DT + 1e-6
        )


def test_scenario_perception_dropout_mid_approach():
    """The blocker scenario: the camera goes blind while closing on a lead."""

    def perception_ok(index: int) -> bool:
        return not (60 <= index < 120)

    history = run_scenario(
        frames=900,
        ego_speed0=12.0,
        lead_gap0=140.0,
        lead_speed_fn=lambda i, t: 0.0,
        perception_ok_fn=perception_ok,
    )
    _assert_no_collision(history, floor_m=0.0)
    _assert_command_envelope(history)

    during = history[60:120]
    assert during[0].state is SafetyState.LIMITED
    assert during[3].state is SafetyState.MIN_RISK_MANEUVER
    for record in during:
        assert record.command.throttle == 0.0, "throttle during a perception dropout"
    for previous, current in zip(during, during[1:]):
        assert current.plan_target_mps <= previous.plan_target_mps + 1e-9
    # The plant's jerk limit means the achieved acceleration takes a few frames to
    # cross zero; after that the speed must fall monotonically.
    settled = during[5:]
    for previous, current in zip(settled, settled[1:]):
        assert current.ego_speed_mps <= previous.ego_speed_mps + 1e-9
    assert during[-1].ego_speed_mps < during[0].ego_speed_mps - 1.0


def test_scenario_sensor_returns_after_a_dropout():
    """After the sensor recovers on a clear road the system must return to NOMINAL."""

    def perception_ok(index: int) -> bool:
        return not (100 <= index < 105)

    history = run_scenario(
        frames=400, ego_speed0=10.0, lead_gap0=None, perception_ok_fn=perception_ok
    )
    _assert_command_envelope(history)
    assert history[100].state is SafetyState.LIMITED
    assert any(record.state is SafetyState.NOMINAL for record in history[110:])
    assert history[-1].state is SafetyState.NOMINAL


def test_scenario_noisy_range_does_not_chatter_the_brake():
    """+/- 1 m of range noise (a few pixels of box height) must not modulate the brake."""
    history = run_scenario(
        frames=800,
        ego_speed0=10.0,
        lead_gap0=32.0,
        lead_speed_fn=lambda i, t: 10.0,
        range_noise_m=1.0,
    )
    _assert_command_envelope(history)
    _assert_no_collision(history, floor_m=0.0)
    for previous, current in zip(history, history[1:]):
        assert abs(current.command.brake - previous.command.brake) <= 0.45, (
            "brake stepped %.2f -> %.2f on frame %d"
            % (previous.command.brake, current.command.brake, current.index)
        )


def test_scenario_measured_jerk_stays_bounded_on_a_clean_run():
    history = run_scenario(frames=600, ego_speed0=5.0, lead_gap0=None)
    _assert_command_envelope(history)
    speeds = [record.ego_speed_mps for record in history]
    accels = [(b - a) / DT for a, b in zip(speeds, speeds[1:])]
    for a, b in zip(accels, accels[1:]):
        assert abs(b - a) / DT <= 3.5 + 1e-6


def test_scenario_full_run_never_leaves_the_envelope():
    rng = random.Random(SEED + 3)
    history = run_scenario(
        frames=1500,
        ego_speed0=rng.uniform(0.0, 15.0),
        lead_gap0=rng.uniform(15.0, 70.0),
        lead_speed_fn=lambda i, t: 8.0 + 6.0 * math.sin(t / 3.0),
        perception_ok_fn=lambda i: rng.random() > 0.02,
        range_noise_m=0.5,
    )
    _assert_command_envelope(history)
    _assert_no_collision(history, floor_m=0.0)
