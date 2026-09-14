"""Tests for the longitudinal/lateral controller.

Beyond the original directional checks these assert the invariants the previous
stateless controller could not hold: bounded pedal rate, bounded jerk, no
throttle/brake co-engagement, no chatter across zero speed error, and an
integrator that cannot wind up while the actuator is saturated.
"""

from __future__ import annotations

import random

import pytest

from adas.control import PIDLikeLongitudinalController
from adas.core.exceptions import ControlError
from adas.core.models import MotionPlan

SEED = 20240913
DT = 0.05


def _plan(target_speed_mps: float = 15.0, steering_angle_deg: float = 0.0) -> MotionPlan:
    return MotionPlan(
        target_speed_mps=target_speed_mps, steering_angle_deg=steering_angle_deg, reason="test"
    )


# --------------------------------------------------------------------------- #
# Original directional behaviour
# --------------------------------------------------------------------------- #


def test_controller_accelerates():
    controller = PIDLikeLongitudinalController(kp_speed=0.15)
    cmd = controller.to_command(_plan(20.0), current_speed_mps=10.0)
    assert cmd.throttle > 0
    assert cmd.brake == 0


def test_controller_brakes():
    controller = PIDLikeLongitudinalController(kp_speed=0.15)
    cmd = controller.to_command(_plan(5.0), current_speed_mps=20.0)
    assert cmd.throttle == 0
    assert cmd.brake > 0


def test_controller_maintains_speed():
    controller = PIDLikeLongitudinalController(kp_speed=0.15)
    cmd = controller.to_command(_plan(15.0), current_speed_mps=15.0)
    assert cmd.throttle < 0.1
    assert cmd.brake < 0.1


def test_controller_steering():
    controller = PIDLikeLongitudinalController(max_steering_angle_deg=25.0)
    right = controller.to_command(_plan(15.0, 10.0), current_speed_mps=15.0)
    assert right.steering > 0
    assert right.steering <= 1.0
    left = controller.to_command(_plan(15.0, -10.0), current_speed_mps=15.0)
    assert left.steering < 0
    assert left.steering >= -1.0


def test_controller_steering_deadband():
    controller = PIDLikeLongitudinalController(steering_deadband_deg=0.5)
    cmd = controller.to_command(_plan(15.0, 0.3), current_speed_mps=15.0)
    assert cmd.steering == 0.0


def test_controller_limits_throttle():
    controller = PIDLikeLongitudinalController(kp_speed=10.0)
    cmd = controller.to_command(_plan(100.0), current_speed_mps=0.0)
    assert 0.0 <= cmd.throttle <= 1.0


def test_controller_saturates_steering_at_full_lock():
    controller = PIDLikeLongitudinalController(max_steering_angle_deg=25.0)
    assert controller.to_command(_plan(15.0, 90.0), 15.0).steering == 1.0
    assert controller.to_command(_plan(15.0, -90.0), 15.0).steering == -1.0


# --------------------------------------------------------------------------- #
# Rate and jerk limits -- ADAS-DEC-11
# --------------------------------------------------------------------------- #


def test_brake_cannot_step_from_zero_to_full_in_one_frame():
    controller = PIDLikeLongitudinalController(kp_speed=1.0)
    cmd = controller.to_command(_plan(0.0), current_speed_mps=30.0, dt_s=DT)
    assert cmd.brake <= controller.brake_apply_rate_per_s * DT + 1e-9
    assert cmd.brake < 1.0


def test_pedal_rates_are_bounded_over_an_adversarial_target_sequence():
    rng = random.Random(SEED)
    controller = PIDLikeLongitudinalController(kp_speed=0.5)
    previous_throttle = 0.0
    previous_brake = 0.0
    speed = 10.0
    for _ in range(4000):
        target = rng.choice([0.0, 30.0, rng.uniform(0.0, 30.0)])
        cmd = controller.to_command(_plan(target), current_speed_mps=speed, dt_s=DT)
        assert cmd.throttle - previous_throttle <= controller.throttle_rate_per_s * DT + 1e-9
        assert cmd.brake - previous_brake <= controller.brake_apply_rate_per_s * DT + 1e-9
        assert previous_brake - cmd.brake <= controller.brake_release_rate_per_s * DT + 1e-9
        previous_throttle = cmd.throttle
        previous_brake = cmd.brake
        speed = max(
            0.0,
            speed
            + (cmd.throttle * controller.accel_authority_mps2
               - cmd.brake * controller.brake_authority_mps2) * DT,
        )


def test_commanded_acceleration_respects_the_jerk_limit():
    """The jerk limit is ONE-SIDED, and this pins both sides of that.

    A RISING deceleration throws an unbraced occupant forward and is what the
    limit exists to bound.  Coming off the brake returns the occupant toward zero
    g against the seat back, and a ceiling on the release rate is a requirement
    to keep braking -- which contradicts the requirement, asserted everywhere
    else in this project, that an unwarranted deceleration be removed promptly.
    The release is therefore bounded by its own, larger limit.
    """
    rng = random.Random(SEED + 1)
    controller = PIDLikeLongitudinalController(kp_speed=1.0)
    previous = controller.commanded_accel_mps2
    for _ in range(2000):
        controller.to_command(
            _plan(rng.uniform(0.0, 30.0)), current_speed_mps=rng.uniform(0.0, 30.0), dt_s=DT
        )
        current = controller.commanded_accel_mps2
        if current < previous:
            # More deceleration: the comfort band.
            assert previous - current <= controller.max_jerk_mps3 * DT + 1e-9
        else:
            # Less deceleration, or more acceleration.
            ceiling = (
                controller.release_jerk_mps3
                if previous < 0.0
                else controller.max_jerk_mps3
            )
            assert current - previous <= ceiling * DT + 1e-9
        previous = current


def test_emergency_relaxes_the_jerk_limit_but_not_the_envelope():
    controller = PIDLikeLongitudinalController(kp_speed=1.0)
    normal = controller.to_command(_plan(0.0), 30.0, dt_s=DT).brake
    controller.reset()
    urgent = controller.to_command(_plan(0.0), 30.0, dt_s=DT, emergency=True).brake
    assert urgent > normal
    assert urgent <= 1.0


def test_throttle_and_brake_are_never_both_engaged():
    rng = random.Random(SEED + 2)
    controller = PIDLikeLongitudinalController(kp_speed=0.8)
    speed = 5.0
    for _ in range(3000):
        cmd = controller.to_command(
            _plan(rng.uniform(0.0, 25.0), rng.uniform(-30.0, 30.0)),
            current_speed_mps=speed,
            dt_s=rng.uniform(0.02, 0.15),
        )
        assert not (cmd.throttle > 0.0 and cmd.brake > 0.0)
        assert 0.0 <= cmd.throttle <= 1.0
        assert 0.0 <= cmd.brake <= 1.0
        assert -1.0 <= cmd.steering <= 1.0
        speed = max(0.0, min(40.0, speed + rng.uniform(-1.0, 1.0)))


# --------------------------------------------------------------------------- #
# Deadband, hysteresis and anti-windup
# --------------------------------------------------------------------------- #


def test_no_actuator_chatter_around_zero_speed_error():
    """Dither the target across the setpoint; the actuators must not alternate."""
    rng = random.Random(SEED + 3)
    controller = PIDLikeLongitudinalController(kp_speed=0.15)
    switches = 0
    previous_mode = "coast"
    for _ in range(600):
        target = 15.0 + rng.uniform(-0.2, 0.2)
        cmd = controller.to_command(_plan(target), current_speed_mps=15.0, dt_s=DT)
        mode = "throttle" if cmd.throttle > 0 else ("brake" if cmd.brake > 0 else "coast")
        if mode != previous_mode and "coast" not in (mode, previous_mode):
            switches += 1
        previous_mode = mode
    assert switches == 0, "throttle/brake alternated %d times inside the deadband" % switches


def test_deadband_produces_no_pedal_action():
    controller = PIDLikeLongitudinalController(speed_deadband_mps=0.3)
    cmd = controller.to_command(_plan(15.2), current_speed_mps=15.0, dt_s=DT)
    assert cmd.throttle == 0.0
    assert cmd.brake == 0.0


def test_integrator_does_not_wind_up_while_saturated():
    controller = PIDLikeLongitudinalController(kp_speed=1.0, ki_speed=0.5)
    for _ in range(500):
        controller.to_command(_plan(30.0), current_speed_mps=0.0, dt_s=DT)
    saturated_integral = controller._integral_mps
    assert abs(controller.ki_speed * saturated_integral) <= controller.integral_limit_mps2 + 1e-9
    # And the controller must still be able to give the pedal back.
    for _ in range(200):
        cmd = controller.to_command(_plan(0.0), current_speed_mps=30.0, dt_s=DT)
    assert cmd.throttle == 0.0
    assert cmd.brake > 0.0


def test_integral_removes_a_steady_state_offset():
    """Constant load: pure P settles with an offset, PI must close it."""
    controller = PIDLikeLongitudinalController(kp_speed=0.4, ki_speed=0.3)
    speed = 10.0
    load_mps2 = 0.4  # constant grade/drag
    for _ in range(4000):
        cmd = controller.to_command(_plan(12.0), current_speed_mps=speed, dt_s=DT)
        accel = (
            cmd.throttle * controller.accel_authority_mps2
            - cmd.brake * controller.brake_authority_mps2
            - load_mps2
        )
        speed = max(0.0, speed + accel * DT)
    assert abs(speed - 12.0) < 0.5


# --------------------------------------------------------------------------- #
# Validation and state
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("speed", [-1.0, float("nan"), float("inf")])
def test_invalid_ego_speed_raises(speed):
    controller = PIDLikeLongitudinalController()
    with pytest.raises(ControlError):
        controller.to_command(_plan(10.0), current_speed_mps=speed)


@pytest.mark.parametrize(
    "plan",
    [
        MotionPlan(float("nan"), 0.0, "bad"),
        MotionPlan(10.0, float("nan"), "bad"),
        MotionPlan(-5.0, 0.0, "bad"),
    ],
)
def test_invalid_plan_raises(plan):
    controller = PIDLikeLongitudinalController()
    with pytest.raises(ControlError):
        controller.to_command(plan, current_speed_mps=10.0)


def test_unusable_dt_falls_back_to_the_nominal_period():
    controller = PIDLikeLongitudinalController(kp_speed=1.0)
    a = controller.to_command(_plan(30.0), 0.0, dt_s=0.0)
    controller.reset()
    b = controller.to_command(_plan(30.0), 0.0, dt_s=0.05)
    assert a.throttle == pytest.approx(b.throttle)


def test_reset_clears_the_rate_limiter_history():
    controller = PIDLikeLongitudinalController(kp_speed=1.0)
    for _ in range(50):
        controller.to_command(_plan(30.0), 0.0, dt_s=DT)
    assert controller.commanded_accel_mps2 > 0.0
    controller.reset()
    assert controller.commanded_accel_mps2 == 0.0
    first = controller.to_command(_plan(30.0), 0.0, dt_s=DT)
    assert first.throttle <= controller.throttle_rate_per_s * DT + 1e-9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kp_speed": 0.0},
        {"ki_speed": -0.1},
        {"max_steering_angle_deg": 0.0},
        {"accel_authority_mps2": 0.0},
        {"max_jerk_mps3": 0.0},
        {"max_jerk_mps3": 10.0, "max_jerk_emergency_mps3": 1.0},
        {"throttle_rate_per_s": 0.0},
        {"emergency_stop_time_s": 0.0},
        {"emergency_stop_time_s": -1.0},
        {"emergency_stop_time_s": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    from adas.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        PIDLikeLongitudinalController(**kwargs)


# --------------------------------------------------------------------------- #
# Emergency authority -- the AEB decision must be executable by this controller
# --------------------------------------------------------------------------- #


def test_emergency_commands_a_real_brake_and_reaches_full_authority():
    """Regression: the controller used to answer an AEB with ~0.06 of brake.

    With kp_speed = 0.15 a proportional speed error of 15 m/s is only 2.25 m/s^2,
    and with the planner's old rate-limited AEB target the error was 0.4 m/s, i.e.
    0.06 m/s^2 -- essentially no brake at all.  The emergency feed-forward makes
    the demand ``|error| / emergency_stop_time_s``, saturated at the brake
    authority, so a real AEB reaches full brake within the emergency jerk limit.
    """
    controller = PIDLikeLongitudinalController()  # shipped gains: kp = 0.15
    speed = 15.0
    brakes = []
    for _ in range(20):
        cmd = controller.to_command(_plan(0.0), current_speed_mps=speed, dt_s=DT, emergency=True)
        brakes.append(cmd.brake)
        assert cmd.throttle == 0.0
    assert max(brakes) >= 0.99, "peak brake was only %.3f" % max(brakes)
    # And it gets there monotonically at no more than the emergency jerk limit.
    step = controller.max_jerk_emergency_mps3 * DT / controller.brake_authority_mps2
    previous = 0.0
    for value in brakes:
        assert value - previous <= step + 1e-9
        previous = value


def test_without_emergency_the_same_error_stays_comfortable():
    """The feed-forward must not leak into normal driving."""
    controller = PIDLikeLongitudinalController()
    brakes = [
        controller.to_command(_plan(0.0), current_speed_mps=15.0, dt_s=DT).brake
        for _ in range(20)
    ]
    step = controller.max_jerk_mps3 * DT / controller.brake_authority_mps2
    previous = 0.0
    for value in brakes:
        assert value - previous <= step + 1e-9
        previous = value
    assert max(brakes) < 0.6, "comfort braking reached %.3f" % max(brakes)


def test_emergency_brake_scales_with_the_remaining_speed_error():
    """The demand is |error| / emergency_stop_time_s, saturated at authority."""
    controller = PIDLikeLongitudinalController(max_jerk_emergency_mps3=1e6)
    cmd = controller.to_command(_plan(0.0), current_speed_mps=2.0, dt_s=DT, emergency=True)
    expected = 2.0 / controller.emergency_stop_time_s / controller.brake_authority_mps2
    assert cmd.brake == pytest.approx(expected, rel=1e-6)
    controller.reset()
    cmd = controller.to_command(_plan(0.0), current_speed_mps=25.0, dt_s=DT, emergency=True)
    assert cmd.brake == pytest.approx(1.0, rel=1e-9), "should saturate at full brake"


def test_emergency_ignores_the_comfort_deadband():
    """There is no comfort requirement during an AEB event."""
    controller = PIDLikeLongitudinalController(speed_deadband_mps=0.3)
    quiet = controller.to_command(_plan(14.8), current_speed_mps=15.0, dt_s=DT)
    assert quiet.brake == 0.0
    controller.reset()
    urgent = controller.to_command(_plan(14.8), current_speed_mps=15.0, dt_s=DT, emergency=True)
    assert urgent.brake > 0.0


def test_emergency_never_commands_throttle():
    """Even asked to speed up, an emergency frame must not open the throttle."""
    rng = random.Random(SEED + 9)
    controller = PIDLikeLongitudinalController()
    for _ in range(2000):
        cmd = controller.to_command(
            _plan(rng.uniform(0.0, 30.0)),
            current_speed_mps=rng.uniform(0.0, 30.0),
            dt_s=DT,
            emergency=True,
        )
        assert cmd.throttle == 0.0


# --------------------------------------------------------------------------- #
# State integrity
# --------------------------------------------------------------------------- #


def test_a_frame_that_raises_does_not_advance_the_rate_limiters():
    """A command the actuator never received must not become the next baseline.

    This pins a property; it is not a regression test for an observed defect.  The
    controller used to mutate its state inside ``_longitudinal`` before validating,
    but ``validate_motion_plan`` runs first and rejects every input found that can
    make a later stage raise, so no sequence was demonstrated that actually moved
    the rate limiters on a rejected frame.  Committing after validation makes the
    property hold by construction rather than by coincidence.
    """
    controller = PIDLikeLongitudinalController()
    good = controller.to_command(_plan(20.0), current_speed_mps=10.0, dt_s=DT)
    assert good.throttle > 0.0
    snapshot = (
        controller._integral_mps,
        controller._prev_accel_mps2,
        controller._prev_throttle,
        controller._prev_brake,
        controller._mode,
    )
    with pytest.raises(ControlError):
        controller.to_command(
            MotionPlan(target_speed_mps=0.0, steering_angle_deg=float("nan"), reason="bad"),
            current_speed_mps=10.0,
            dt_s=DT,
        )
    assert (
        controller._integral_mps,
        controller._prev_accel_mps2,
        controller._prev_throttle,
        controller._prev_brake,
        controller._mode,
    ) == snapshot, "state advanced on a frame whose command was never issued"


def test_integrator_always_unwinds_when_the_error_reverses():
    """Clamping anti-windup must never latch the integrator at its bound.

    Also a pinned property rather than a proven defect: with these limits the
    integrator cannot be pushed past its clamp in the first place, so the plain
    scheme was not shown to latch.  The explicit unwind clause removes the
    dependence on that coincidence.
    """
    controller = PIDLikeLongitudinalController(kp_speed=0.2, ki_speed=0.3)
    # A small, persistent error: large enough to accumulate, small enough that the
    # acceleration command never saturates, so the integrator reaches its clamp.
    for _ in range(400):
        controller.to_command(_plan(12.0), current_speed_mps=10.0, dt_s=DT)
    wound = controller._integral_mps
    assert wound > 0.0, "setup failed: the integrator never wound up"
    assert abs(controller.ki_speed * wound) == pytest.approx(
        controller.integral_limit_mps2, rel=0.05
    ), "setup failed: the integrator did not reach its clamp"
    previous = abs(wound)
    crossed = False
    for _ in range(400):
        controller.to_command(_plan(8.0), current_speed_mps=10.0, dt_s=DT)
        value = controller._integral_mps
        if value <= 0.0:
            crossed = True
            break
        assert abs(value) < previous + 1e-12, "integrator did not shrink under an opposing error"
        previous = abs(value)
    assert crossed, (
        "integrator never unwound past zero; it latched at %.4f" % controller._integral_mps
    )


# --------------------------------------------------------------------------- #
# AEB authority: the controller must be able to brake on its own
#
# Regression for the blocker "the planner+controller cannot execute an AEB; the
# arbiter is the only thing that brakes hard enough to avoid a collision". Two
# independent attenuators produced it and both are pinned here:
#   (1) the emergency jerk limit was 15 m/s^3, so full authority took 0.53 s;
#   (2) there was no target-rate feed-forward, so a plan asking for 3 m/s^2 of
#       comfort deceleration produced ~0.4 m/s^2 of demand and the ego closed on
#       the lead while the planner was already asking it to stop.
# --------------------------------------------------------------------------- #


def test_emergency_reaches_full_authority_inside_the_jerk_ceiling():
    """A held AEB target of 0 must saturate the brake at the emergency jerk rate.

    The bound was 250 ms, which the controller met by ramping at 40 m/s^3 -- a
    figure taken from the brake actuator alone and twice what the safety
    specification permits.  20 m/s^3 is the ceiling, it reaches 8 m/s^2 in the
    0.4 s a human panic brake takes, and braking faster buys no stopping distance
    because the brake's own 0.15 s rise time filters it out.  So the requirement
    is that the controller uses ALL of the authorised rate and none of the
    unauthorised: full pedal in 8 frames, and not one frame later.
    """
    controller = PIDLikeLongitudinalController()
    brakes = []
    for _ in range(12):
        cmd = controller.to_command(_plan(0.0), current_speed_mps=15.0, dt_s=DT, emergency=True)
        brakes.append(cmd.brake)
    first_full = next(i for i, b in enumerate(brakes) if b >= 0.99)
    expected = controller.brake_authority_mps2 / controller.max_jerk_emergency_mps3
    assert first_full * DT <= expected + DT + 1e-9, "full brake only at %.2f s: %s" % (
        first_full * DT, brakes
    )
    assert brakes == sorted(brakes), "emergency brake must not back off while ramping"
    assert all(b == 0.0 for b in [
        controller.to_command(_plan(0.0), current_speed_mps=15.0, dt_s=DT, emergency=True).throttle
    ])


def test_emergency_ramp_is_still_jerk_limited():
    """Fast is not instant: the first emergency frame must respect the jerk limit."""
    controller = PIDLikeLongitudinalController()
    cmd = controller.to_command(_plan(0.0), current_speed_mps=15.0, dt_s=DT, emergency=True)
    ceiling = controller.max_jerk_emergency_mps3 * DT / controller.brake_authority_mps2
    assert 0.0 < cmd.brake <= ceiling + 1e-9


def test_comfort_ramp_is_followed_not_trailed():
    """A target falling at 3 m/s^2 must produce ~3 m/s^2 of demand, not 0.4."""
    controller = PIDLikeLongitudinalController()
    target = 15.0
    for _ in range(40):
        target -= 3.0 * DT
        cmd = controller.to_command(_plan(target), current_speed_mps=15.0, dt_s=DT)
    # 3.0 asked for, minus the soft feed-forward deadband, plus whatever the PI
    # contributes from the (by now large) speed error -- so at least 2.5 m/s^2.
    assert cmd.brake * controller.brake_authority_mps2 >= 2.5
    assert cmd.throttle == 0.0


def test_feed_forward_ignores_target_noise():
    """Target dither must not reach the pedals through the feed-forward."""
    rng = random.Random(SEED + 11)
    controller = PIDLikeLongitudinalController()
    acted = 0
    for _ in range(400):
        cmd = controller.to_command(
            _plan(15.0 + rng.uniform(-0.2, 0.2)), current_speed_mps=15.0, dt_s=DT
        )
        acted += int(cmd.brake > 0.0 or cmd.throttle > 0.0)
    assert acted == 0, "%d frames of pedal action from pure target noise" % acted


def test_target_step_is_not_smeared_across_the_window():
    """After a target step the feed-forward must clear, not ring for N frames.

    The brake still walks back down under the comfort jerk limit -- that is the
    actuator model, not the feed-forward -- so what is pinned here is that it
    only ever DECREASES after the step and reaches zero. A window still holding
    the pre-step targets would re-brake part way through the tail.
    """
    controller = PIDLikeLongitudinalController()
    for _ in range(20):
        controller.to_command(_plan(15.0), current_speed_mps=15.0, dt_s=DT)
    step = controller.to_command(
        _plan(0.0), current_speed_mps=15.0, dt_s=DT, emergency=True
    ).brake
    tail = [
        controller.to_command(_plan(15.0), current_speed_mps=15.0, dt_s=DT).brake
        for _ in range(3 * controller.target_ff_window)
    ]
    assert max(tail) <= step + 1e-9, "brake rose again after the step: %s" % tail
    assert tail == sorted(tail, reverse=True), "brake not monotone after step: %s" % tail
    assert tail[-1] == 0.0, "feed-forward still braking after the step: %s" % tail


def test_feed_forward_cannot_exceed_brake_authority():
    """An absurd target ramp is clamped by the authority, not by luck."""
    controller = PIDLikeLongitudinalController()
    target = 40.0
    for _ in range(30):
        target = max(0.0, target - 20.0 * DT)
        cmd = controller.to_command(_plan(target), current_speed_mps=40.0, dt_s=DT)
        assert 0.0 <= cmd.brake <= 1.0
        assert cmd.throttle == 0.0


def test_reset_clears_the_feed_forward_window():
    controller = PIDLikeLongitudinalController()
    target = 15.0
    for _ in range(20):
        target -= 3.0 * DT
        controller.to_command(_plan(target), current_speed_mps=15.0, dt_s=DT)
    assert controller.to_command(_plan(target), current_speed_mps=15.0, dt_s=DT).brake > 0.0
    controller.reset()
    cmd = controller.to_command(_plan(target), current_speed_mps=target, dt_s=DT)
    assert cmd.brake == 0.0 and cmd.throttle == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_ff_window": 2},
        {"target_ff_deadband_mps2": -0.1},
        {"target_ff_step_mps": 0.0},
        {"target_ff_step_mps": float("nan")},
    ],
)
def test_feed_forward_tuning_is_validated(kwargs):
    with pytest.raises(Exception):
        PIDLikeLongitudinalController(**kwargs)


def test_emergency_stop_is_held_to_standstill():
    """No comfort deadband during an AEB: the brake must not let the car creep.

    The emergency demand decays with the remaining speed, so with the comfort
    pedal deadband still applied the brake was released at 0.043 m/s and the ego
    coasted the last few centimetres into the obstacle.
    """
    controller = PIDLikeLongitudinalController()
    speed = 15.0
    released_while_moving = []
    for _ in range(2000):
        cmd = controller.to_command(_plan(0.0), current_speed_mps=speed, dt_s=DT, emergency=True)
        if speed > 1e-3 and cmd.brake == 0.0:
            released_while_moving.append(round(speed, 4))
        speed = max(0.0, speed - 8.0 * cmd.brake * DT)
        if speed <= 1e-3:
            break
    assert not released_while_moving, (
        "brake released at %s m/s during an emergency stop" % released_while_moving[:5]
    )
    assert speed <= 1e-3, "never reached standstill, stuck at %.4f m/s" % speed
