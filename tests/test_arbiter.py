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


NOMINAL_FOCAL_PX = (WIDTH / 2.0) / math.tan(math.radians(30.0))
"""Focal length implied by a 60 deg horizontal field of view, as the arbiter
assumes when the caller supplies none."""


def _track(
    track_id: int = 1,
    distance_m: float = 30.0,
    velocity_mps: float = 0.0,
    center_x: float = WIDTH / 2.0,
    **kwargs,
) -> TrackedObject:
    """A track whose box and metric lateral offset are GEOMETRICALLY CONSISTENT.

    The fixture used to leave ``lateral_offset_m`` at its 0.0 default while
    putting the box anywhere in the image, which is a physically impossible
    object: a box 580 px left of centre at 1.5 m cannot also be dead ahead.  It
    did not matter while the in-path gate was purely image-space; it matters now
    that the gate is metric first, because the two halves of the fixture
    disagreed about where the object was.  The offset is therefore derived from
    the box unless the caller states one, which is what a real tracker does.
    """
    if "lateral_offset_m" not in kwargs and distance_m > 0.0:
        kwargs["lateral_offset_m"] = (center_x - WIDTH / 2.0) * distance_m / NOMINAL_FOCAL_PX
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


def _approach(
    arbiter,
    gap0: float,
    closing_mps: float,
    ego_speed_mps: float,
    frames: int,
    command: Optional[ControlCommand] = None,
    plan: Optional[MotionPlan] = None,
    track_id: int = 1,
    dt: float = DT,
    **track_kwargs,
):
    """Feed ``frames`` frames of a closing approach and return every result.

    The arbiter measures its own closing rate from a window of RAW ranges, so a
    single hand-built frame carries no rate at all and -- deliberately -- earns no
    braking.  Anything that asks what the arbiter DOES about a closure has to show
    it one.
    """
    out = []
    gap = gap0
    cmd = command or ControlCommand(0.4, 0.0, 0.0)
    mp = plan or MotionPlan(ego_speed_mps, 0.0, "cruise")
    for index in range(frames):
        track = _track(track_id=track_id, distance_m=max(0.1, gap), **track_kwargs)
        track.hits = index + 1
        track.age_frames = index + 1
        out.append(
            arbiter.arbitrate(mp, cmd, _context(
                ego_speed_mps, tracks=[track], dt_s=dt, timestamp_s=index * dt
            ))
        )
        gap -= closing_mps * dt
    return out


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
        vetoed = any("vetoed_empty_road" in v for v in result.violations) or (
            "brake_vetoed_no_hazard" in arbiter.last_shaping
        )
        if vetoed:
            # THE ONE EXCEPTION, and it is the difference between an authority and
            # a wire: healthy perception, a valid moving ego, and no object
            # anywhere in the arbiter's own corridor.  A brake arriving from
            # upstream then rests on no measurement any part of this system can
            # make, and an unwarranted full-authority stop on a motorway does not
            # avoid a collision, it manufactures one behind.
            assert arbiter.last_lead is None
            assert result.command.brake <= cmd.brake + 1e-9
            assert result.command.brake <= arbiter.limits.no_hazard_decel_cap_mps2 / 8.0 + 1e-9
        else:
            assert result.command.brake >= cmd.brake - 1e-9, (
                "the arbiter attenuated a brake: %.3f -> %.3f"
                % (cmd.brake, result.command.brake)
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

    The floor only ever holds an inherited brake on for longer, so it cannot
    attenuate anything.  A lead is present throughout so that the veto -- the one
    path that IS allowed to lower a brake -- is not what is being measured.
    """
    limits = _limits(brake_release_rate_per_s=8.0)
    arbiter = SafetyArbiter(limits)
    lead = _track(distance_m=60.0)
    arbiter.arbitrate(
        MotionPlan(0.0, 0.0, "x"), ControlCommand(0.0, 1.0, 0.0), _context(10.0, tracks=[lead])
    )
    result = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"), ControlCommand(0.0, 0.0, 0.0), _context(10.0, tracks=[lead])
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


def test_the_arbiter_reports_the_jerk_of_its_own_output_not_of_the_wheel_speed():
    """What the arbiter checks is the quantity it CONTROLS.

    The previous version differentiated the measured ego speed twice and filed
    the result as a violation.  That is a statement about the vehicle and its
    wheel-speed sensor, not about the system's output: +/-0.05 m/s of bus noise
    is +/-40 m/s^3 of apparent jerk, larger than every limit in the module, and
    on a vehicle answering a perfectly legal 20 m/s^3 demand through a 0.15 s
    brake rise it reported the vehicle for obeying.  The rate of change of the
    DEMAND is what the specification bounds and what this component sets, so it
    is what is checked -- which means the report can only fire on a step that
    arrived from upstream, and that is exactly the defect worth reporting.
    """
    arbiter = SafetyArbiter()
    lead = _track(distance_m=60.0)
    plan = MotionPlan(10.0, 0.0, "ok")
    arbiter.arbitrate(plan, ControlCommand(0.0, 0.0, 0.0), _context(10.0, tracks=[lead]))
    stepped = arbiter.arbitrate(
        plan, ControlCommand(0.0, 1.0, 0.0), _context(10.0, tracks=[lead])
    )
    assert any("jerk_" in v for v in stepped.violations), stepped.violations

    # And a demand built at the emergency rate is not reported, because it is legal.
    smooth = SafetyArbiter()
    results = _approach(smooth, gap0=36.0, closing_mps=20.0, ego_speed_mps=20.0, frames=40)
    assert max(r.command.brake for r in results) > 0.9, "the setup never braked hard"
    assert not any(
        "jerk_" in v for r in results for v in r.violations
    ), "the arbiter reported its own compliant ramp"


# --------------------------------------------------------------------------- #
# Independence -- ADAS-DEC-13
# --------------------------------------------------------------------------- #


def test_arbiter_brakes_on_a_hazard_the_plan_denies():
    """Plan says cruise at full throttle; the raw ranges say closing fast."""
    arbiter = SafetyArbiter()
    results = _approach(
        arbiter,
        gap0=30.0,
        closing_mps=15.0,
        ego_speed_mps=15.0,
        frames=14,
        command=ControlCommand(1.0, 0.0, 0.0),
        plan=MotionPlan(15.0, 0.0, "cruise_clear"),
    )
    last = results[-1]
    assert last.command.throttle == 0.0
    assert last.command.brake > 0.3
    assert last.state is SafetyState.MIN_RISK_MANEUVER


def test_the_arbiter_adds_no_braking_on_a_tracks_first_frame():
    """The founding defect, stated as the property that removes it.

    A track the arbiter has never seen carries no closing rate.  The only prior
    available is "assume it is stationary in the world", ``rate = -v_ego``, and
    letting the emergency tests read that prior braked a 20 m/s ego to a
    standstill behind a lead holding a constant 32.5 m.  The prior is gone: on
    frame 0 the arbiter reports that it has not measured a rate and adds no brake
    of its own.  What it may still do on the measured RANGE alone is refuse
    throttle, which is a state, not an actuation.
    """
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "cruise_clear"),
        ControlCommand(0.8, 0.0, 0.0),
        _context(15.0, tracks=[_track(distance_m=15.0)]),
    )
    assert arbiter.last_lead is not None
    assert arbiter.last_lead.rate_is_measured is False
    assert arbiter.last_lead.range_rate_mps == 0.0, "a rate was fabricated"
    assert arbiter.last_lead.required_decel_mps2 == 0.0
    assert result.command.brake == 0.0


def test_arbiter_ignores_tracker_velocity_mps():
    """A slanderous tracker velocity must leave no trace at all.

    ``TrackedObject.velocity_mps`` is an unfiltered reciprocal derivative the
    tracker computed from the same range the arbiter is given, so reading it
    would be believing one channel twice.  1230 m/s of claimed closing speed
    changes nothing, in either direction.
    """
    def run(claimed):
        arbiter = SafetyArbiter()
        results = _approach(
            arbiter, gap0=60.0, closing_mps=0.0, ego_speed_mps=10.0, frames=25,
            command=ControlCommand(0.2, 0.0, 0.0), velocity_mps=claimed,
        )
        return [r.command.brake for r in results], [r.state for r in results]

    slander, slander_states = run(1230.0)
    honest, honest_states = run(0.0)
    reversed_, _ = run(-1230.0)
    assert slander == honest == reversed_
    assert all(state is SafetyState.NOMINAL for state in slander_states)
    assert max(slander) == 0.0


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


def test_range_channel_disagreement_degrades_and_keeps_the_pinhole_every_frame():
    """A disagreeing second channel degrades the state and is never adopted.

    A monocular depth model that under-reads range used to drive the AEB directly,
    with no plausibility test: one frame of ``depth = 15 m`` against a 40 m pinhole
    was enough.

    The rule that replaced it is latch-free, so the answer must be the SAME on
    every frame rather than changing on the third. Two ranges that differ by more
    than ``range_disagreement_frac`` are a channel fault, not a second opinion: the
    pinhole is used, the disagreement is reported, and the state is degraded --
    frame 1 and frame 20 alike. The previous rule reported the same fault for
    three frames and then adopted the disagreeing channel anyway, which is the
    opposite conclusion drawn from identical evidence.
    """
    arbiter = SafetyArbiter(_limits())
    for index in range(20):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"), ControlCommand(0.2, 0.0, 0.0), _disagreeing_context(15.0)
        )
        assert any("range_channel_disagreement" in v for v in result.violations), index
        assert not any("uncorroborated" in v for v in result.violations), index
        assert result.state is SafetyState.LIMITED, index
        assert arbiter.last_lead.distance_m == pytest.approx(40.0, abs=1.0), index
        assert arbiter.last_lead.source is RangeSource.PINHOLE, index


def test_an_agreeing_nearer_second_channel_is_adopted_on_its_first_frame():
    """Corroboration cuts the other way too: agreement is believed at once.

    A depth estimate 8% nearer than the pinhole is inside
    ``range_disagreement_frac``, so both channels are credible and the arbiter
    takes the SHORTER of the two -- on the first frame, with no dwell. A backstop
    that waited three frames to believe the nearer of two agreeing ranges would be
    spending its margin on a distinction neither measurement supports.
    """
    arbiter = SafetyArbiter(_limits())
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(
            15.0,
            tracks=[_track(distance_m=40.0)],
            independent_ranges={
                1: RangeEstimate(
                    distance_m=37.0, confidence=0.9, source=RangeSource.DEPTH_MODEL
                )
            },
        ),
    )
    assert arbiter.last_lead.distance_m == pytest.approx(37.0)
    assert arbiter.last_lead.source is RangeSource.FUSED
    assert not any("disagreement" in v for v in result.violations)


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


def test_a_range_teleport_re_anchors_and_returns_the_rate_to_unmeasured():
    """The 68 m -> 6.8 m association teleport used to imply 1230 m/s of closing.

    What the arbiter does with it now is neither of the two things it has done
    before.  It does not differentiate it -- that is the 1230 m/s fantasy -- and
    it does not re-seed the rate at ``-ego_speed`` either, because that is a
    fabricated number and this arbiter does not act on fabricated numbers.  It
    drops the window: the rate goes back to UNMEASURED and stays there until
    fresh captures rebuild it, which is the honest description of what the
    arbiter knows about an object whose identity has just been re-anchored.
    """
    arbiter = SafetyArbiter()
    for index in range(8):
        arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"),
            ControlCommand(0.2, 0.0, 0.0),
            _context(15.0, tracks=[_track(distance_m=68.3)], timestamp_s=index * DT),
        )
    assert arbiter.last_lead.rate_is_measured is True, "precondition: a rate existed"
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(15.0, tracks=[_track(distance_m=6.8)], timestamp_s=8 * DT),
    )
    assert any("range_jump" in v for v in result.violations)
    assert arbiter.last_lead.rate_is_measured is False
    assert arbiter.last_lead.range_rate_mps == 0.0, "the jump was differentiated"
    assert result.command.brake == 0.0, "a re-anchor is not a closure"
    # The RANGE, though, is a measurement, and 6.8 m at 15 m/s is inside half the
    # speed-matched RSS gap: the arbiter refuses throttle and says why.
    assert result.command.throttle == 0.0
    assert result.state is SafetyState.LIMITED


def test_a_teleport_to_a_farther_range_does_not_brake():
    arbiter = SafetyArbiter()
    for index in range(8):
        arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"),
            ControlCommand(0.2, 0.0, 0.0),
            _context(15.0, tracks=[_track(distance_m=30.0)], timestamp_s=index * DT),
        )
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(15.0, tracks=[_track(distance_m=95.0)], timestamp_s=8 * DT),
    )
    assert any("range_jump" in v for v in result.violations)
    assert arbiter.last_lead.rate_is_measured is False
    assert result.command.brake == 0.0


def test_out_of_path_object_is_not_a_lead():
    """A car one lane over is 3.5 m to the side and is not in the way."""
    arbiter = SafetyArbiter()
    next_lane = _track(track_id=5, distance_m=12.0, lateral_offset_m=3.5)
    result = arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"),
        ControlCommand(0.2, 0.0, 0.0),
        _context(10.0, tracks=[next_lane], ego_lane_half_width_frac=0.2),
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
    """``min_in_path_half_width_frac`` is a floor on the IMAGE fallback corridor.

    The image corridor is what the arbiter falls back on when perception supplies
    no metric lateral offset for an object, so these tests pass ``None`` for it.
    """
    arbiter = SafetyArbiter(_limits(min_in_path_half_width_frac=0.30))
    # 0.30 of 1280 = 384 px each side of centre: the corridor is [256, 1024].
    just_inside = _track(
        track_id=1, distance_m=8.0, center_x=1010.0, lateral_offset_m=None
    )
    arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"),
        ControlCommand(0.3, 0.0, 0.0),
        # The planner would have used a 0.05 band; the arbiter must not.
        _context(10.0, tracks=[just_inside], ego_lane_half_width_frac=0.05),
    )
    assert arbiter.last_lead is not None

    wider = SafetyArbiter(_limits(min_in_path_half_width_frac=0.30))
    outside_both = _track(
        track_id=1, distance_m=8.0, center_x=1250.0, lateral_offset_m=None
    )
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
    straddling = _track(
        track_id=1, distance_m=8.0, center_x=1080.0, lateral_offset_m=None
    )
    arbiter.arbitrate(
        MotionPlan(10.0, 0.0, "x"), ControlCommand(0.3, 0.0, 0.0), _context(10.0, tracks=[straddling])
    )
    assert arbiter.last_lead is not None


def test_a_confident_lane_widens_the_corridor_and_never_moves_it():
    """A trusted lane centre adds objects to the image fallback; never removes them."""
    def off_centre():
        return _track(track_id=1, distance_m=8.0, center_x=1150.0, lateral_offset_m=None)

    without = SafetyArbiter()
    without.arbitrate(
        MotionPlan(10.0, 0.0, "x"), ControlCommand(0.3, 0.0, 0.0),
        _context(10.0, tracks=[off_centre()]),
    )
    assert without.last_lead is None

    with_lane = SafetyArbiter()
    with_lane.arbitrate(
        MotionPlan(10.0, 0.0, "x"),
        ControlCommand(0.3, 0.0, 0.0),
        _context(10.0, tracks=[off_centre()], lane=_lane(1100.0, confidence=0.9, is_mock=False)),
    )
    assert with_lane.last_lead is not None


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
    """A SUSTAINED over-acceleration is reported; a single noisy frame is not.

    The check differentiates the vehicle bus, and a bus with noise on it produces
    single-frame excursions that say nothing about the control.  The estimate is
    a least-squares slope over the evidence window and an exceedance has to
    persist for ``kinematics_streak_frames`` before it is filed.
    """
    limits = _limits(max_acceleration_mps2=3.0, kinematics_streak_frames=3)
    arbiter = SafetyArbiter(limits)
    speed = 10.0
    reported = False
    for index in range(20):
        result = arbiter.arbitrate(
            MotionPlan(speed, 0.0, "x"), ControlCommand(0.1, 0.0, 0.0),
            _context(speed, tracks=[_track(distance_m=90.0)], timestamp_s=index * DT),
        )
        reported = reported or any("measured_accel" in v for v in result.violations)
        speed += 6.0 * DT  # 6 m/s^2, twice the limit, sustained
    assert reported

    calm = SafetyArbiter(limits)
    speed = 10.0
    for index in range(20):
        quiet = calm.arbitrate(
            MotionPlan(speed, 0.0, "x"), ControlCommand(0.1, 0.0, 0.0),
            _context(speed, tracks=[_track(distance_m=90.0)], timestamp_s=index * DT),
        )
        assert not any("measured_accel" in v for v in quiet.violations)
        speed += 1.0 * DT


def test_unjustified_hard_deceleration_is_flagged_but_a_hazard_stop_is_not():
    limits = _limits(max_deceleration_mps2=8.0, kinematics_streak_frames=3)
    arbiter = SafetyArbiter(limits)
    speed = 30.0
    seen = []
    for index in range(10):
        unjustified = arbiter.arbitrate(
            MotionPlan(speed, 0.0, "x"), ControlCommand(0.0, 0.1, 0.0),
            _context(speed, tracks=[_track(distance_m=90.0)], timestamp_s=index * DT),
        )
        seen.extend(unjustified.violations)
        speed = max(0.0, speed - 12.0 * DT)  # 12 m/s^2, above the vehicle's authority
    assert any(
        "measured_decel" in v and "unjustified" in v for v in seen
    ), seen

    # The same deceleration, with a hazard the arbiter can see, is the arbiter's
    # own doing and must not be filed against it.
    hazard_arbiter = SafetyArbiter(limits)
    gap = 25.0
    speed = 20.0
    findings = []
    for index in range(20):
        track = _track(track_id=1, distance_m=max(0.5, gap))
        track.hits = index + 1
        result = hazard_arbiter.arbitrate(
            MotionPlan(20.0, 0.0, "x"), ControlCommand(0.0, 0.5, 0.0),
            _context(speed, tracks=[track], timestamp_s=index * DT),
        )
        findings.extend(v for v in result.violations if "unjustified" in v)
        gap -= speed * DT
        speed = max(0.0, speed - 12.0 * DT)
    assert not findings, findings


# --------------------------------------------------------------------------- #
# Degradation state machine -- ADAS-DEC-04
# --------------------------------------------------------------------------- #


def test_one_perception_dropout_enters_limited():
    """A single blink degrades the state -- and does NOT start braking.

    ``limited_after_dropouts`` is 1 because a frame with no picture is a real
    degradation and the throttle must come off for it.  ``blind_hold_frames`` is
    8 because a dropped frame is a dropped frame: the vehicle holds what it was
    doing.  The two used to be conflated, and a 0.15 s blink declared a
    minimum-risk manoeuvre with its throttle lock, its steering hold and its
    ten-frame recovery -- a degraded state that outlived every cause it had.
    """
    arbiter = SafetyArbiter()
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "x"),
        ControlCommand(0.8, 0.0, 0.0),
        _context(15.0, perception_ok=False),
    )
    assert result.state is SafetyState.LIMITED
    assert result.command.throttle == 0.0
    assert result.command.brake == 0.0, "one blink is not a reason to brake"


def test_half_a_second_of_blindness_enters_a_minimum_risk_manoeuvre():
    """0.4 s at 20 m/s is 8 m travelled without a picture: that is a stop."""
    arbiter = SafetyArbiter()
    limits = arbiter.limits
    states = []
    brakes = []
    for index in range(limits.blind_hold_frames + 12):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"),
            ControlCommand(0.8, 0.0, 0.0),
            _context(15.0, perception_ok=False, timestamp_s=index * DT),
        )
        states.append(result.state)
        brakes.append(result.command.brake)
    assert states[0] is SafetyState.LIMITED
    assert states[-1] is SafetyState.MIN_RISK_MANEUVER
    assert brakes[-1] > 0.0
    # A minimum-risk stop is a CONTROLLED stop.  Nothing was detected in front of
    # the vehicle -- the reason to stop is that it cannot see -- and the traffic
    # behind has no reason to expect an AEB.
    assert max(brakes) <= limits.mrm_decel_mps2 / limits.brake_authority_mps2 + 1e-9
    assert limits.mrm_decel_mps2 < limits.emergency_grade_mps2


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
    """DISENGAGE needs a fault that braking cannot survive -- not a blind camera.

    A missing ego state is one: none of the hazard maths is defined without it,
    and no amount of braking makes it come back.  A perception loss is NOT one;
    see the test below.
    """
    limits = _limits(disengage_after_frames=5)
    arbiter = SafetyArbiter(limits)
    blind_ego = SafetyContext(
        ego=None,
        tracks=[],
        perception=PerceptionStatus(ok=True),
        dt_s=DT,
        frame_width_px=WIDTH,
        frame_height_px=HEIGHT,
    )
    for _ in range(6):
        arbiter.arbitrate(MotionPlan(15.0, 0.0, "x"), ControlCommand(0.5, 0.0, 0.0), blind_ego)
    assert arbiter.state is SafetyState.DISENGAGE
    for _ in range(100):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"), ControlCommand(0.5, 0.0, 0.0), _context(15.0)
        )
        assert result.state is SafetyState.DISENGAGE
        assert result.command.throttle == 0.0
    arbiter.reset()
    assert arbiter.state is SafetyState.NOMINAL


def test_a_perception_loss_never_disengages_a_moving_vehicle():
    """R7.  DISENGAGE hands a moving vehicle back to nobody.

    There is no driver in this loop.  A loss of perception -- however prolonged
    -- is answered by a controlled stop, and the controlled stop is what the
    vehicle does for as long as the loss lasts.  The fault is still reported
    every frame; what it may not do is accumulate toward a hand-back.
    """
    arbiter = SafetyArbiter(_limits(disengage_after_frames=5))
    for index in range(400):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "x"),
            ControlCommand(0.5, 0.0, 0.0),
            _context(15.0, perception_ok=False, timestamp_s=index * DT),
        )
        assert result.state is not SafetyState.DISENGAGE, "frame %d" % index
    assert any("perception_dropout" in v for v in result.violations)
    assert result.state is SafetyState.MIN_RISK_MANEUVER
    assert result.command.brake > 0.0


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
    """The counter still exists; what feeds it is now only the terminal class."""
    limits = _limits(disengage_after_frames=4)
    arbiter = SafetyArbiter(limits)
    for index in range(limits.disengage_after_frames):
        result = arbiter.arbitrate(
            None,  # no plan at all: the decision path is gone
            ControlCommand(0.5, 0.0, 0.0),
            _context(15.0, timestamp_s=index * DT),
        )
    assert result.state is SafetyState.DISENGAGE
    assert result.command.throttle == 0.0
    assert result.command.brake > 0.0


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


def test_a_stationary_obstacle_is_braked_for_as_soon_as_it_is_MEASURED():
    """The repro (/tmp/rev_cutin.py frame 0), and the price of not guessing.

    Three designs have met on this scenario and the test pins the third.

    * A new track's rate started at ZERO -- the optimistic prior -- so a
      stationary car 15 m ahead with the ego at 15 m/s reported ``ttc=inf`` and
      commanded brake 0.000.  Blind exactly when a hazard appeared.
    * Seeding the rate at ``-ego_speed`` fixed that and then let the emergency
      tests act on the fabricated number, which braked at full authority for a
      lead holding a constant 32.5 m.
    * Now: no prior at all.  Frame 0 carries one range measurement, which is not
      a rate, so the arbiter adds no braking and says so.  The closure exists at
      the third distinct capture and full authority follows the jerk ramp.

    The cost of the wait is bounded by physics rather than by policy, and this is
    where it is written down: at 55 ms of sense latency on a 50 ms grid no closing
    rate can exist before decision frame 2 whatever the code does, so the two
    frames below are the floor and not a concession.
    """
    arbiter = SafetyArbiter()
    frames = []
    distance = 15.0
    speed = 15.0
    for index in range(12):
        track = _track(track_id=1, distance_m=distance)
        track.hits = index + 1
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "cruise_clear"),
            ControlCommand(0.8, 0.0, 0.0),
            _context(speed, tracks=[track], timestamp_s=index * DT),
        )
        frames.append((result, arbiter.last_lead))
        distance -= speed * DT  # the obstacle is stationary; the ego closes on it

    first, lead = frames[0]
    assert lead is not None
    assert lead.rate_is_measured is False, "one sample is not a measurement"
    assert lead.range_rate_mps == 0.0, "a rate was fabricated"
    assert first.command.brake == 0.0
    # 15 m at 15 m/s is OUTSIDE half the speed-matched RSS gap (6.84 m), so the
    # range alone is not a hazard either and the arbiter adds nothing at all on
    # frame 0.  That is the correct answer: reacting here would be reacting to a
    # measurement the system has not made.  The primary path's own following law
    # is what carries these frames, and the harness's closed-loop scenarios are
    # what check that it does.
    assert first.state is SafetyState.NOMINAL

    measured = next(i for i, (_, l) in enumerate(frames) if l.rate_is_measured)
    assert measured <= 3, "the closure took %d frames to measure" % measured
    aeb_frame = next(
        i for i, (r, _) in enumerate(frames) if r.state is SafetyState.MIN_RISK_MANEUVER
    )
    assert aeb_frame <= measured + 2, "full authority took %d frames" % aeb_frame
    assert frames[-1][0].command.brake > 0.9


def test_a_lead_that_keeps_pace_converges_off_the_seed_and_returns_to_nominal():
    """The safe prior must be transient, not a permanent phantom brake.

    The range is 20 m, INSIDE the 26 m radius in which the seed used to trip a
    full-authority AEB on frame 0 at this speed. The previous version of this test
    used 45 m, which is outside that radius, so it asserted nothing about the
    failure it is named for.
    """
    arbiter = SafetyArbiter()
    result = None
    states = []
    peak_brake = 0.0
    for index in range(40):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "follow"),
            ControlCommand(0.2, 0.0, 0.0),
            _context(15.0, tracks=[_track(track_id=1, distance_m=20.0)], timestamp_s=index * DT),
        )
        states.append(result.state)
        peak_brake = max(peak_brake, result.command.brake)
    assert SafetyState.MIN_RISK_MANEUVER not in states, "phantom AEB on a constant range"
    assert peak_brake <= 3.0 / 8.0 + 1e-9, "more than the comfort rate: %.3f" % peak_brake
    assert abs(arbiter.last_lead.range_rate_mps) < 0.2
    assert arbiter.last_lead.rate_is_measured is True
    assert result.state is SafetyState.NOMINAL
    assert result.command.brake == 0.0


# --------------------------------------------------------------------------- #
# Minimum-risk manoeuvre lateral behaviour
# --------------------------------------------------------------------------- #


def test_an_mrm_holds_the_steering_it_last_commanded_instead_of_straightening():
    """An MRM brakes to a stop on the CURRENT path; it does not straighten."""
    limits = _limits(
        max_steering_rate_rad_s=10.0, mrm_straighten_speed_mps=1.0, mrm_after_dropouts=3,
        blind_hold_frames=2,
    )
    arbiter = SafetyArbiter(limits)
    turning = ControlCommand(0.2, 0.0, 0.25)
    for index in range(10):
        nominal = arbiter.arbitrate(
            MotionPlan(8.0, 6.0, "lane_center"), turning, _context(8.0, timestamp_s=index * DT)
        )
    held = nominal.command.steering
    assert held > 0.05, "the setup never actually commanded a steering angle"

    # Perception drops out: MRM. The steering must stay on the path.
    for index in range(4):
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
    limits = _limits(
        max_steering_rate_rad_s=10.0, mrm_straighten_speed_mps=1.0, mrm_after_dropouts=2,
        blind_hold_frames=1,
    )
    arbiter = SafetyArbiter(limits)
    for index in range(6):
        turning = arbiter.arbitrate(
            MotionPlan(5.0, 0.0, "x"),
            ControlCommand(0.2, 0.0, 0.2),
            _context(5.0, timestamp_s=index * DT),
        )
    assert turning.command.steering == pytest.approx(0.2)

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
    # The arithmetic, rather than a round number: the lead brakes at 6 m/s^2
    # from 12 m/s and travels 12 m, so the ego has 36 + 12 - 2 = 46 m to stop
    # from 12 m/s and 144 / (2 x 46) = 1.57 m/s^2 does it.  A stack that reached
    # 2.4 m/s^2 here would be over-braking by half again, which this project
    # measures as transferring the collision to the vehicle behind.
    peak = max(record.command.brake for record in history) * 8.0
    assert 1.5 <= peak <= 1.57 * 1.5 + 0.5, "peak deceleration %.2f m/s^2" % peak
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
    # The cut-in frame carries ONE range sample, which is not a closing rate, and
    # 20 m at 15 m/s is outside the range-only hazard boundary too.  There is
    # nothing measured on that frame to cut the throttle for, and cutting it
    # anyway is the stationary-prior reflex this redesign removed.  What is
    # required is that the response follows the measurement promptly: the closure
    # exists at the third distinct capture and the throttle is gone by then.
    cut = next(
        (i for i, r in enumerate(history[40:60], start=40) if r.command.throttle == 0.0),
        None,
    )
    assert cut is not None and cut <= 45, "throttle still on at frame %s" % cut
    # The state stays NOMINAL and that is the correct answer, which is worth
    # saying out loud because the previous design degraded here.  Holding 2.25 m
    # of clearance against a 5 m/s closure at 20 m needs 25 / (2 x 17.75) =
    # 0.70 m/s^2 -- a quarter of the comfort limit.  The following law has this
    # situation; declaring a degradation for it would be the system announcing it
    # had given up on an ordinary lane change.
    assert all(record.command.throttle == 0.0 for record in history[cut:60])
    assert max(r.command.brake for r in history[40:120]) * 8.0 < 3.0, (
        "a 0.7 m/s^2 requirement was answered with more than comfort braking"
    )


def test_scenario_lead_disappears_does_not_step_the_throttle():
    """Losing a track must not produce a throttle or target-speed step."""

    def present(index: int) -> bool:
        return index < 200

    history = run_scenario(
        frames=600, ego_speed0=8.0, lead_gap0=25.0, lead_speed_fn=lambda i, t: 8.0,
        track_present_fn=present,
    )
    _assert_command_envelope(history)
    # The plan TARGET is a setpoint, not a trajectory: when the lead disappears
    # the planner asks for cruise again in one step, and the rate at which the
    # vehicle approaches it is the controller's business.  What must not step is
    # the PEDAL, which is what the actuator sees.
    for previous, current in zip(history, history[1:]):
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
    # A dropped frame is a dropped frame; half a second of nothing is a vehicle
    # driving blind, and 0.4 s at 12 m/s is 5 m travelled without a picture.
    blind_hold = SafetyArbiter().limits.blind_hold_frames
    assert during[blind_hold].state is SafetyState.MIN_RISK_MANEUVER
    for record in during:
        assert record.command.throttle == 0.0, "throttle during a perception dropout"
    for previous, current in zip(during, during[1:]):
        assert current.plan_target_mps <= previous.plan_target_mps + 1e-9
    # The plant's jerk limit means the achieved acceleration takes a few frames to
    # cross zero; after that the speed must fall monotonically.
    settled = during[blind_hold + 5:]
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


# --------------------------------------------------------------------------- #
# The phantom AEB -- a fabricated rate may inform caution, never full authority
# --------------------------------------------------------------------------- #


def _matched_rss_m(limits: ArbiterLimits, speed_mps: float) -> float:
    """RSS gap for a lead travelling at exactly the ego speed -- no rate at all."""
    return max(
        limits.absolute_min_gap_m,
        speed_mps * limits.reaction_time_s
        + speed_mps ** 2 / (2.0 * limits.ego_brake_capability_mps2)
        - speed_mps ** 2 / (2.0 * limits.lead_brake_capability_mps2),
    )


@pytest.mark.parametrize(
    "speed_mps, old_phantom_radius_m", [(10.0, 13.0), (15.0, 26.0), (20.0, 43.0), (25.0, 66.0)]
)
def test_no_emergency_authority_at_all_on_a_tracks_first_frame(
    speed_mps, old_phantom_radius_m
):
    """/tmp/hunt2.py, as a permanent test, and the boundary has moved to zero.

    The threshold sweep that found the blocker: on the FIRST frame of a brand-new
    track -- which is also what a track-id change, an occlusion exit and a range
    jump look like -- the arbiter used to authorise a full-authority stop for any
    in-path object closer than 13 m at 10 m/s, 26 m at 15 m/s, 43 m at 20 m/s and
    66 m at 25 m/s, whatever the object was actually doing.  At motorway speed
    that is inside a normal two-second following distance.

    The fix then was to bound the radius by half the speed-matched RSS gap.  The
    fix now is that there is no radius: an emergency stop requires a MEASURED
    closure, one range sample is not one, and no distance makes it one.  What a
    short range still does -- and this is the rung of authority a measurement of
    the RANGE alone supports -- is degrade the state and lock out the throttle.
    """
    limits = ArbiterLimits()
    boundary = limits.aeb_headway_frac * _matched_rss_m(limits, speed_mps)
    assert boundary < old_phantom_radius_m / 2.0, "the test is not proving anything"

    for distance_m in (boundary * 0.5, boundary * 0.9, boundary * 1.1,
                       old_phantom_radius_m * 0.9):
        arbiter = SafetyArbiter()
        result = arbiter.arbitrate(
            MotionPlan(speed_mps, 0.0, "cruise"),
            ControlCommand(0.5, 0.0, 0.0),
            _context(speed_mps, tracks=[_track(track_id=1, distance_m=distance_m)]),
        )
        assert result.state is not SafetyState.MIN_RISK_MANEUVER, (
            "ego %.0f m/s, new track at %.1f m: %s" % (speed_mps, distance_m, result.state.value)
        )
        assert result.command.brake == 0.0, (
            "ego %.0f m/s, new track at %.1f m: braked %.3f on one range sample"
            % (speed_mps, distance_m, result.command.brake)
        )
        # Inside the range-only boundary the throttle still comes off, because the
        # RANGE is a measurement even when the rate is not.
        if distance_m < boundary:
            assert result.state is SafetyState.LIMITED
            assert result.command.throttle == 0.0


@pytest.mark.parametrize("speed_mps, distance_m", [(10.0, 12.0), (15.0, 25.0), (20.0, 40.0), (25.0, 60.0)])
def test_a_lead_at_a_constant_range_is_never_emergency_braked(speed_mps, distance_m):
    """/tmp/hunt9.py, as a permanent test.

    A lead holding a CONSTANT range is not closing, at any speed and at any range.
    The seed said otherwise on the first frame of every (re-)initialisation, and
    the closed-loop consequence was an ego braked from 20.00 m/s to 0.00 m/s in
    5.7 s behind a car that never moved.
    """
    arbiter = SafetyArbiter()
    states = []
    peak = 0.0
    braked_frames = 0
    for index in range(60):
        result = arbiter.arbitrate(
            MotionPlan(speed_mps, 0.0, "follow"),
            ControlCommand(0.4, 0.0, 0.0),
            _context(
                speed_mps,
                tracks=[_track(track_id=1, distance_m=distance_m)],
                timestamp_s=index * DT,
            ),
        )
        states.append(result.state)
        peak = max(peak, result.command.brake)
        if result.command.brake > 1e-6:
            braked_frames += 1
    assert SafetyState.MIN_RISK_MANEUVER not in states, "phantom AEB at %.0f m" % distance_m
    ceiling = arbiter.limits.deferred_aeb_decel_mps2 / arbiter.limits.brake_authority_mps2
    assert peak <= ceiling + 1e-9, "peak brake %.3f on a constant range" % peak
    assert states[-1] is SafetyState.NOMINAL, "never recovered: %s" % states[-1].value
    # The transient must be a PULSE, not a stop: the safe prior only holds until
    # there is a measurement, which is the rate window plus a frame of brake-release
    # shaping. Before the fix the brake never came off at all.
    assert braked_frames <= 8, "%d of 60 frames braked on a constant range" % braked_frames


def test_a_re_identified_track_does_not_re_arm_the_phantom():
    """A track-id change is the commonest re-initialisation on real footage.

    /tmp/hunt8.py: through the real tracker at 20 m/s behind a lead at a constant
    32.5 m, the arbiter fired a full 8 m/s^2 stop on the frame the track appeared,
    again after a 0.45 s detector miss, and again when the track was re-identified.
    """
    arbiter = SafetyArbiter()
    peak = 0.0
    saw_mrm = False
    for index in range(60):
        track_id = 1 if index < 20 else (2 if index < 40 else 3)  # two re-ids
        result = arbiter.arbitrate(
            MotionPlan(20.0, 0.0, "follow"),
            ControlCommand(0.4, 0.0, 0.0),
            _context(
                20.0,
                tracks=[_track(track_id=track_id, distance_m=32.5)],
                timestamp_s=index * DT,
            ),
        )
        peak = max(peak, result.command.brake)
        saw_mrm = saw_mrm or result.state is SafetyState.MIN_RISK_MANEUVER
    assert not saw_mrm, "a re-id re-armed the phantom AEB"
    assert peak <= 3.0 / 8.0 + 1e-9, "peak brake %.3f across two re-ids" % peak



# --------------------------------------------------------------------------- #
# Range-source stability
# --------------------------------------------------------------------------- #


def test_a_flip_flopping_range_source_neither_reseeds_nor_brakes():
    """The reviewer's frames 166-169 of /tmp/vidrun.log, as a permanent test.

    The measured range was flat at 7.0-7.4 m, the fusion decision flipped
    pinhole<->fused on every single frame, each flip called
    ``_range_filter.forget()``, and the re-seeded ``-ego_speed`` rate held
    ``brake = 1.00`` on a lead that was not moving relative to the ego.
    """
    arbiter = SafetyArbiter()
    switches = 0
    peak = 0.0
    saw_mrm = False
    measured_once = False
    for index in range(80):
        # The second channel is present on even frames and absent on odd ones --
        # the exact provenance dither the real depth channel produces.
        ranges = (
            {1: RangeEstimate(distance_m=25.0, confidence=0.8, source=RangeSource.DEPTH_MODEL)}
            if index % 2 == 0
            else None
        )
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "follow"),
            ControlCommand(0.3, 0.0, 0.0),
            _context(
                15.0,
                tracks=[_track(track_id=1, distance_m=25.0)],
                independent_ranges=ranges,
                timestamp_s=index * DT,
            ),
        )
        switches += sum(1 for v in result.violations if "range_source_switch" in v)
        peak = max(peak, result.command.brake)
        saw_mrm = saw_mrm or result.state is SafetyState.MIN_RISK_MANEUVER
        # The opening frames legitimately carry the safe prior. What must never
        # happen is the prior coming BACK: a source change re-anchors the filter,
        # it does not discard and re-seed it.
        if arbiter.last_lead.rate_is_measured:
            measured_once = True
        assert not (measured_once and not arbiter.last_lead.rate_is_measured), (
            "frame %d threw the measured rate away and went back to the seed" % index
        )
        if measured_once:
            assert abs(arbiter.last_lead.range_rate_mps) < 2.0, (
                "frame %d: %.1f m/s on a flat range"
                % (index, arbiter.last_lead.range_rate_mps)
            )
    assert not saw_mrm, "the provenance dither produced an emergency stop"
    assert peak <= 3.0 / 8.0 + 1e-9, "peak brake %.3f on a flat range" % peak
    assert switches <= 2, "%d reported source switches over 80 frames" % switches
    assert arbiter.last_lead.rate_is_measured is True
    assert abs(arbiter.last_lead.range_rate_mps) < 0.5


def test_a_dithering_second_channel_confidence_does_not_toggle_the_source():
    """Hysteresis plus dwell on the confidence gate, not a bare threshold."""
    arbiter = SafetyArbiter()
    switches = 0
    for index in range(60):
        confidence = 0.36 if index % 2 == 0 else 0.34  # straddling min_range_confidence
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "follow"),
            ControlCommand(0.3, 0.0, 0.0),
            _context(
                15.0,
                tracks=[_track(track_id=1, distance_m=30.0)],
                independent_ranges={
                    1: RangeEstimate(
                        distance_m=30.0, confidence=confidence, source=RangeSource.DEPTH_MODEL
                    )
                },
                timestamp_s=index * DT,
            ),
        )
        switches += sum(1 for v in result.violations if "range_source_switch" in v)
    assert switches == 0, "%d source switches from a dithering confidence" % switches


def test_a_non_finite_second_channel_confidence_is_discarded_not_propagated():
    """/tmp/hunt3.py section J, as a permanent test.

    ``nan < 0.35`` is False, so a NaN confidence passed the low-confidence gate,
    NaN-weighted the blend and produced a NaN fused range. Every hazard comparison
    against NaN is False, so a car 8 m dead ahead at 15 m/s produced
    ``state=nominal brake=0.000 lead d=nan`` and the input throttle went straight
    through to the actuators.
    """
    arbiter = SafetyArbiter()
    distance = 8.0
    first = None
    result = None
    for index in range(14):
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "cruise"),
            ControlCommand(0.5, 0.0, 0.0),
            _context(
                15.0,
                tracks=[_track(track_id=1, distance_m=distance)],
                independent_ranges={
                    1: RangeEstimate(
                        distance_m=distance,
                        confidence=float("nan"),
                        source=RangeSource.DEPTH_MODEL,
                    )
                },
                timestamp_s=index * DT,
            ),
        )
        if index == 0:
            first = result
            lead = arbiter.last_lead
            assert lead is not None
            assert math.isfinite(lead.distance_m), "a NaN confidence blinded the arbiter"
            assert lead.distance_m == pytest.approx(8.0)
            assert lead.source is RangeSource.PINHOLE
        distance = max(0.5, distance - 15.0 * DT)  # the obstacle is stationary
    assert any("confidence_not_finite" in v for v in first.violations)
    assert first.state is not SafetyState.NOMINAL
    assert first.command.throttle == 0.0
    # ... and once the closure is measured it is a full-authority stop, not a NaN.
    assert result.state is SafetyState.MIN_RISK_MANEUVER
    assert result.command.brake > 0.9


# --------------------------------------------------------------------------- #
# A perception dropout must not erase the hazard being braked for
# --------------------------------------------------------------------------- #


def test_a_dropout_keeps_assessing_the_coasted_lead_instead_of_forgetting_it():
    """/tmp/hunt4.py, as a permanent test.

    ``arbitrate()`` guarded the whole lead assessment with ``if perception.ok``.
    The tracker coasts its tracks through a blink, so the arbiter was handed a
    populated track list on exactly those frames and threw it away: the command
    collapsed from the AEB demand to the flat 3.5 m/s^2 MRM floor for the duration
    of the dropout, while still closing on the obstacle, and jumped straight back
    the frame perception returned.
    """
    arbiter = SafetyArbiter()
    speed = 15.0
    distance = 25.0
    brakes = []
    for index in range(20):
        ok = not (10 <= index < 15)
        coast = 0 if ok else index - 9
        result = arbiter.arbitrate(
            MotionPlan(15.0, 0.0, "follow"),
            ControlCommand(0.0, 0.0, 0.0),
            _context(
                speed,
                tracks=[_track(track_id=1, distance_m=distance, time_since_update=coast)],
                perception_ok=ok,
                timestamp_s=index * DT,
            ),
        )
        brakes.append(result.command.brake)
        distance = max(1.0, distance - speed * DT)
    before = brakes[9]
    during = brakes[10:15]
    assert before > 0.5, "the setup never established an emergency brake: %.3f" % before
    for offset, value in enumerate(during):
        assert value >= before - 1e-9, (
            "dropout frame %d cut the brake %.3f -> %.3f" % (10 + offset, before, value)
        )
    assert min(during) > 3.5 / 8.0 + 0.05, "the brake fell back to the flat MRM floor"


def test_a_coasted_range_cannot_by_itself_authorise_an_emergency_stop():
    """An extrapolation is not a measurement.

    A track that has been coasting since its first frame has no MEASURED range
    history at all, so the rate-dependent AEB tests stay inhibited; only the
    rate-independent ones (which read the coasted range, not a rate) can fire.
    """
    arbiter = SafetyArbiter()
    distance = 18.0
    for index in range(5):
        result = arbiter.arbitrate(
            MotionPlan(20.0, 0.0, "follow"),
            ControlCommand(0.0, 0.0, 0.0),
            _context(
                20.0,
                tracks=[_track(track_id=1, distance_m=distance, time_since_update=index + 1)],
                perception_ok=False,
                timestamp_s=index * DT,
            ),
        )
        distance -= 20.0 * DT
    assert arbiter.last_lead is not None
    assert arbiter.last_lead.coasting is True
    assert arbiter.last_lead.rate_is_measured is False, "an extrapolation became a measurement"
    assert result.state is not SafetyState.MIN_RISK_MANEUVER, (
        "an extrapolated range authorised an emergency stop"
    )
    assert result.command.throttle == 0.0, "throttle survived a perception dropout"
    # The track is still ASSESSED -- forgetting it is the defect the coast window
    # exists to prevent -- it simply supports no braking of its own.
    assert arbiter.last_lead.distance_m == pytest.approx(18.0 - 4 * 20.0 * DT)


def test_a_track_coasted_past_the_limit_is_dropped_rather_than_believed():
    arbiter = SafetyArbiter()
    limits = arbiter.limits
    result = arbiter.arbitrate(
        MotionPlan(15.0, 0.0, "follow"),
        ControlCommand(0.0, 0.0, 0.0),
        _context(
            15.0,
            tracks=[
                _track(
                    track_id=1,
                    distance_m=6.0,
                    time_since_update=limits.max_coast_frames + 1,
                )
            ],
            perception_ok=False,
        ),
    )
    assert arbiter.last_lead is None
    assert any("too_stale" in v for v in result.violations)


# --------------------------------------------------------------------------- #
# The arbiter must not treat its own intervention as evidence for itself
# --------------------------------------------------------------------------- #


def test_an_mrm_releases_once_its_own_braking_is_the_only_remaining_finding():
    """/tmp/hunt10.py and the 300-frame closed loop, as a permanent test.

    The arbiter's own MRM braking opens a gap between the planner's target speed
    and the measured ego speed; ``_check_plan`` turned that into
    ``plan_accel_X_above_3.00``, which forced LIMITED, which reset the recovery
    streak, so ``_latch`` could never de-escalate. The vehicle was braked to a
    standstill on an open road. The plan target here is deliberately held at
    20 m/s while the ego is slowed, which is exactly that confound.
    """
    arbiter = SafetyArbiter()
    speed = 20.0
    states = []
    saw_induced = False
    gap = 30.0
    for index in range(160):
        # A real, MEASURED closing hazard for the first 30 frames, then clear road.
        tracks = []
        if index < 30:
            track = _track(track_id=1, distance_m=max(1.0, gap))
            track.hits = index + 1
            tracks = [track]
            gap -= speed * DT
        result = arbiter.arbitrate(
            MotionPlan(20.0, 0.0, "cruise"),
            ControlCommand(0.4, 0.0, 0.0),
            _context(speed, tracks=tracks, timestamp_s=index * DT),
        )
        states.append(result.state)
        saw_induced = saw_induced or any("arbiter_induced" in v for v in result.violations)
        speed = max(0.5, speed - 8.0 * result.command.brake * DT + 2.5 * result.command.throttle * DT)
    assert SafetyState.MIN_RISK_MANEUVER in states, "the real hazard never produced an MRM"
    tail = states[-60:]
    assert SafetyState.MIN_RISK_MANEUVER not in tail, "still braking on an open road"
    assert SafetyState.NOMINAL in tail, "never de-escalated: %s" % sorted({s.value for s in tail})
    assert speed > 5.0, "the arbiter braked the vehicle to a crawl on an open road: %.2f" % speed


def test_the_arbiters_own_deceleration_and_jerk_do_not_hold_it_degraded():
    """The same principle on the kinematics channel.

    A vehicle answering the arbiter's own emergency brake produces a deceleration
    and a jerk spike. Counting those as unrecovered findings resets the recovery
    streak every frame for as long as the arbiter keeps braking.
    """
    arbiter = SafetyArbiter()
    speed = 20.0
    induced = []
    final = None
    gap = 24.0
    for index in range(80):
        tracks = []
        if index < 20:
            track = _track(track_id=1, distance_m=max(1.0, gap))
            track.hits = index + 1
            tracks = [track]
            gap -= max(0.0, speed) * DT
        # The plan target tracks the ego speed, so plan_accel cannot contribute and
        # this test isolates the achieved-deceleration and jerk channel.
        result = arbiter.arbitrate(
            MotionPlan(speed, 0.0, "cruise"),
            ControlCommand(0.4, 0.0, 0.0),
            _context(speed, tracks=tracks, timestamp_s=index * DT),
        )
        induced.extend(
            v for v in result.violations if "arbiter_commanded" in v or "jerk_" in v
        )
        # A deliberately violent plant so the jerk and decel checks trip.
        speed = max(
            0.6, speed - 12.0 * result.command.brake * DT + 2.5 * result.command.throttle * DT
        )
        final = result
    assert induced, "the setup never produced a decel/jerk finding, so nothing was proved"
    assert final.state is SafetyState.NOMINAL, "held degraded by its own braking"
    assert arbiter.state is SafetyState.NOMINAL


def test_a_held_mrm_steering_finding_does_not_hold_the_mrm_open():
    """The lateral instance of the same class.

    In MIN_RISK_MANEUVER the arbiter replaces the planner's steering with its own
    hold, so a ``steering_rate`` or ``lateral_accel`` finding about the discarded
    request is a report, not a reason to stay in the manoeuvre.
    """
    limits = _limits(
        max_steering_rate_rad_s=0.5, recovery_frames=5, mrm_after_dropouts=3,
        blind_hold_frames=2,
    )
    arbiter = SafetyArbiter(limits)
    # Enough dropouts to put the arbiter in MRM; the planner saws at the wheel.
    for index in range(4):
        arbiter.arbitrate(
            MotionPlan(10.0, 0.0, "x"),
            ControlCommand(0.0, 0.0, 0.0),
            _context(10.0, perception_ok=False, timestamp_s=index * DT),
        )
    assert arbiter.state is SafetyState.MIN_RISK_MANEUVER
    result = None
    for index in range(4, 30):
        result = arbiter.arbitrate(
            MotionPlan(10.0, 0.0, "x"),
            ControlCommand(0.0, 0.0, 1.0 if index % 2 else -1.0),
            _context(10.0, timestamp_s=index * DT),
        )
    assert any("steering_rate" in v for v in result.violations), "the setup raised nothing"
    assert result.state is not SafetyState.MIN_RISK_MANEUVER, (
        "the arbiter's own steering hold kept the manoeuvre alive: %s" % result.violations
    )


# --------------------------------------------------------------------------- #
# The arbiter may never make the output LESS safe
# --------------------------------------------------------------------------- #


def test_an_incoming_brake_is_passed_through_unattenuated_with_a_lead_in_view():
    """A benign lead in view must not buy the arbiter a licence to reduce a brake.

    This is the defect that disqualified one of the three redesign candidates and
    that no test in this repository covered. That candidate's veto ceiling was
    armed whenever a lead was merely MEASURED rather than only on a positively
    empty road: handed ``brake = 1.00`` with an ordinary car 45 m ahead at a
    matched speed, it emitted ``0.12`` and logged ``brake_vetoed_8.0_to_1.5``. The
    round before it attenuated 1.00 to 0.25 and drove into the lead.

    The rule is one sentence and it is asymmetric on purpose: **the arbiter may
    always ADD brake and may SUBTRACT one only on a road it positively sees is
    empty.** A lead at 45 m is the opposite of an empty road, so the incoming
    command must reach the actuators intact whatever the arbiter thinks of it.

    The harness misses this: its only stuck-brake case
    (``arbiter_vetoes_stuck_brake_on_empty_road``) uses an empty road, and the
    sweep never hands the arbiter a brake at all.
    """
    arbiter = SafetyArbiter()
    result = None
    for index in range(20):
        result = arbiter.arbitrate(
            MotionPlan(20.0, 0.0, "cruise"),
            ControlCommand(0.0, 1.0, 0.0),
            _context(20.0, tracks=[_track(distance_m=45.0)], timestamp_s=index * DT),
        )
    assert result.command.brake == pytest.approx(1.0), (
        "a brake was attenuated with a lead in view: %.3f, violations %s"
        % (result.command.brake, result.violations)
    )
    assert not any("veto" in v for v in result.violations), result.violations


def test_only_a_positively_empty_road_may_lower_an_incoming_brake():
    """The converse, so the test above cannot be satisfied by never vetoing.

    Same stuck ``brake = 1.00``, no object anywhere. Here the veto is not merely
    permitted, it is required: 8.0 m/s^2 for nothing on a motorway does not avoid
    a collision, it manufactures one behind.
    """
    arbiter = SafetyArbiter()
    result = None
    for index in range(20):
        result = arbiter.arbitrate(
            MotionPlan(20.0, 0.0, "cruise"),
            ControlCommand(0.0, 1.0, 0.0),
            _context(20.0, tracks=[], timestamp_s=index * DT),
        )
    assert result.command.brake * 8.0 < 3.0, (
        "an unwarranted full brake survived on an empty road: %.3f" % result.command.brake
    )


def test_the_arbiters_own_braking_cannot_inflate_its_lead_acceleration_credit():
    """The one place the arbiter's command feeds an input to the arbiter's command.

    ``closure(a_ego)`` adds the ego's measured acceleration back when it recovers
    the lead's acceleration from the curvature of the range, because
    ``d2(range)/dt2 = a_lead - a_ego``. The physics is a DECORRELATION -- the ego
    term cancels exactly -- but it is still the arbiter's own brake reaching an
    input to the arbiter's own brake, which is the shape of every self-sustaining
    loop in this module's history, so it gets a test rather than a docstring.

    Two vehicles at a rigidly constant 30 m gap while the EGO brakes hard at
    -4 m/s^2. The range is constant, so its curvature is zero, so ``a_lead`` is
    ``0 + a_ego``... which would be a credited -4 m/s^2 lead deceleration
    manufactured entirely out of the ego's own brake, and the lead-is-stopping
    branch of the requirement scales with the EGO's speed squared. The sign must
    be the other way: ``a_rel = a_lead - a_ego``, so recovering ``a_lead`` means
    ``a_rel + a_ego``, and with the ego braking the ONLY way the range stays
    constant is that the lead is braking equally hard.
    """
    from adas.control.evidence import EvidenceBook, EvidenceLimits

    book = EvidenceBook(EvidenceLimits())
    track = book.track(1)

    # Case A: constant range, ego braking at -4 m/s^2. The lead must be braking
    # at -4 m/s^2 too, and the credit must say so.
    t = 0.0
    speed = 25.0
    for _ in range(12):
        book.ego_accel_mps2(t, speed)
        track.update(t, 30.0, capture_token=int(t * 1000) + 1)
        t += DT
        speed -= 4.0 * DT
    a_ego = book.ego_accel_mps2(t, speed)
    assert a_ego == pytest.approx(-4.0, abs=0.2), a_ego
    credit = track.closure(a_ego).lead_accel_mps2
    assert credit == pytest.approx(-4.0, abs=0.5), (
        "a constant range while the ego brakes at -4 means the lead brakes at -4; "
        "the credit said %.2f" % credit
    )

    # Case B: the loop. Same ego brake, but the range OPENS at exactly the rate
    # the ego's own deceleration opens it -- i.e. the lead is not braking at all.
    # The credit must be zero, or the arbiter's brake is feeding its own brake.
    book2 = EvidenceBook(EvidenceLimits())
    track2 = book2.track(1)
    t = 0.0
    speed = 25.0
    rng = 30.0
    for _ in range(12):
        book2.ego_accel_mps2(t, speed)
        track2.update(t, rng, capture_token=int(t * 1000) + 1)
        # lead holds 25 m/s; ego is slowing, so the gap opens by (25 - speed) * dt
        rng += (25.0 - speed) * DT
        t += DT
        speed -= 4.0 * DT
    a_ego2 = book2.ego_accel_mps2(t, speed)
    credit2 = track2.closure(a_ego2).lead_accel_mps2
    assert credit2 == pytest.approx(0.0, abs=0.5), (
        "the ego's own -4 m/s^2 was credited to the lead: %.2f" % credit2
    )


# --------------------------------------------------------------------------- #
# A sustained degraded state must not produce one log line per frame
# --------------------------------------------------------------------------- #


def test_a_sustained_degraded_state_logs_transitions_not_frames(caplog):
    """1200 frames of one unchanging fault must not produce 1200 lines.

    This defect has shipped twice. A degraded condition that is a steady state --
    a vehicle with no CAN bus, a stopped camera, a lead held at a short gap in a
    queue -- emitted one WARNING per frame: 1200 lines in a 1200-frame run,
    measured, and 72,000 an hour at 20 Hz. It fills the disk and it buries the
    transition that mattered.

    What must survive the fix is the transition. The entry into the state is
    logged on the frame it happens; the exit is logged on the frame it happens;
    only the unchanged repeat in between is throttled, and the line that gets
    through carries the count of suppressed frames so a throttled condition can
    never read as a quiet one.
    """
    arbiter = SafetyArbiter(_limits(log_repeat_period_s=1e6))  # never repeat
    frames = 1200
    with caplog.at_level(logging.WARNING, logger="adas.control.arbiter"):
        for index in range(frames):
            arbiter.arbitrate(
                MotionPlan(10.0, 0.0, "x"),
                ControlCommand(0.0, 0.0, 0.0),
                _context(10.0, perception_ok=False, timestamp_s=index * DT),
            )
    lines = [r for r in caplog.records if r.name == "adas.control.arbiter"]
    assert 0 < len(lines) <= 6, (
        "%d WARNING lines for %d frames of one unchanging condition; the "
        "transitions are: %s" % (len(lines), frames, [r.getMessage()[:60] for r in lines[:8]])
    )


def test_every_state_transition_is_logged_even_when_the_repeat_is_throttled():
    """The throttle may hide repeats; it may never hide a change of state."""
    arbiter = SafetyArbiter(_limits(log_repeat_period_s=1e6, mrm_after_dropouts=3))
    seen = []
    with caplog_capture(seen):
        for index in range(20):  # blind: nominal -> limited -> min_risk_maneuver
            arbiter.arbitrate(
                MotionPlan(10.0, 0.0, "x"),
                ControlCommand(0.0, 0.0, 0.0),
                _context(10.0, perception_ok=False, timestamp_s=index * DT),
            )
        for index in range(20, 60):  # healthy again: back to nominal
            arbiter.arbitrate(
                MotionPlan(10.0, 0.0, "x"),
                ControlCommand(0.0, 0.0, 0.0),
                _context(10.0, timestamp_s=index * DT),
            )
    states = [m for m in seen if "limited" in m or "min_risk_maneuver" in m or "nominal" in m]
    assert any("limited" in m for m in states), states[:5]
    assert any("min_risk_maneuver" in m for m in states), states[:5]
    assert any("recovered to nominal" in m for m in states), states[-5:]


import contextlib
import logging


@contextlib.contextmanager
def caplog_capture(sink):
    """Collect ``adas.control.arbiter`` messages into ``sink`` at any level."""

    class _H(logging.Handler):
        def emit(self, record):
            sink.append(record.getMessage())

    log = logging.getLogger("adas.control.arbiter")
    handler = _H()
    previous = log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        yield
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)


def test_unused_limits_is_truthful_in_both_directions():
    """``unused_limits()`` must name every inert field and no live one.

    A limit a deployment can set and that nothing reads is worse than an absent
    one: it reads as control and delivers none. Six are currently inert, kept so
    an existing ``SafetyConfig`` still loads, and ``unused_limits()`` is the
    method a start-up check prints. A method that says which fields are dead is
    itself something that goes stale, so it is checked by source inspection of
    the module rather than by a hand-maintained expectation -- which is the same
    technique, and for the same reason, as the harness's own
    ``expectation_field_reads``.

    The check counts reads of ``lim.<name>`` and ``self.limits.<name>`` OUTSIDE
    the ``ArbiterLimits`` class body, so a field's own declaration, docstring and
    ``__post_init__`` range check do not count as the decision reading it.

    ``PROJECTED`` names the fields that reach the decision by being folded into
    ``ArbiterLimits.evidence`` in ``__post_init__`` rather than by being read.
    They are live -- changing one changes behaviour -- and listing them here is
    the one hand-maintained part of this test, so each carries the expression
    that consumes it.

    Written after the test caught five real ones on its first run:
    ``accel_authority_mps2``, ``max_jerk_mps3``, ``max_jerk_emergency_mps3``,
    ``plan_horizon_s`` and ``standstill_gap_m`` were all settable, all
    documented as doing something, and all inert. Two still described an
    achieved-jerk check the redesign had replaced with a command-jerk one.
    """
    PROJECTED = {
        "aeb_min_rate_samples": "evidence.min_samples",
        "aeb_min_rate_span_s": "evidence.min_span_s",
        "range_rate_window_s": "evidence.window_samples",
    }
    import inspect
    import re

    from adas.control import arbiter as mod

    source = inspect.getsource(mod)
    limits_body = inspect.getsource(ArbiterLimits)
    decision = source.replace(limits_body, "")

    declared = {f.name for f in dataclasses.fields(ArbiterLimits)}
    reported_unused = set(ArbiterLimits().unused_limits())
    assert reported_unused <= declared, reported_unused - declared

    def is_read(name):
        return re.search(r"\blim\.%s\b|\bself\.limits\.%s\b" % (name, name), decision) is not None

    lying_dead = sorted(n for n in reported_unused if is_read(n))
    assert not lying_dead, (
        "unused_limits() calls these dead, but the decision reads them: %s" % lying_dead
    )

    for name, target in PROJECTED.items():
        assert name in declared, name
        assert name not in reported_unused, (
            "%s is projected onto %s and IS live; unused_limits() must not call "
            "it dead" % (name, target)
        )

    lying_live = sorted(
        n for n in declared - reported_unused - set(PROJECTED) if not is_read(n)
    )
    assert not lying_live, (
        "these limits are declared and never read, and unused_limits() does not "
        "say so, so a deployment setting one is told nothing: %s" % lying_live
    )


import dataclasses
