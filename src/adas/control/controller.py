"""Low-level control: motion plan -> actuator command.

Units
-----
``target_speed_mps`` / ``current_speed_mps``   m/s
``kp_speed`` / ``ki_speed``                    gains that produce an ACCELERATION
                                               from a speed error, so their units
                                               are 1/s and 1/s^2 respectively.
``accel_authority_mps2``                       m/s^2 delivered at ``throttle = 1``.
``brake_authority_mps2``                       m/s^2 delivered at ``brake = 1``.
``throttle`` / ``brake``                       dimensionless [0, 1].
``steering``                                   dimensionless [-1, 1], where 1.0 is
                                               ``max_steering_angle_deg`` of road
                                               wheel.

Why this controller is stateful
-------------------------------
The previous implementation was a stateless proportional map, which made a rate
limit impossible by construction: a 0.02 -> 1.00 brake step inside one 50 ms frame
was legal.  This version keeps five pieces of state -- the speed-error integral,
the previous commanded acceleration, the previous throttle, the previous brake
and the previous target speed --
and the previous target speed, and uses them to enforce, in order:

1. a symmetric speed-error deadband with hysteresis, so throttle and brake cannot
   alternate frame to frame around zero error;
2. a PI law with clamping anti-windup (the integrator is frozen whenever the
   acceleration command is saturated, or while inside the deadband -- except that
   an update which moves the integrator back TOWARD zero is always accepted, so a
   frozen integrator can never latch);
3. a TARGET-RATE FEED-FORWARD: the planner's target speed is itself rate-limited
   (at ``max_decel_mps2`` on comfort frames, ``mrm_decel_mps2`` on a controlled
   stop), so a plan that says "decelerate at 3 m/s^2" arrives as a target that
   falls 0.15 m/s per 50 ms frame.  A pure error feedback law cannot follow a
   ramp without a standing lag: with ``kp_speed = 0.15 1/s`` it needs a 20 m/s
   speed error to produce 3 m/s^2, so the demand it actually produced while
   trailing a comfort ramp was ~0.4 m/s^2 and the ego kept closing on the lead.
   The controller therefore differentiates the target and adds ``dv*/dt``
   directly to the acceleration demand; the PI law is left to remove the
   residual error only.  The feed-forward is bounded by the actuator authority
   and, on every non-emergency frame, by the planner's own target rate limit, so
   it can never exceed the deceleration the planner asked for;
4. an EMERGENCY FEED-FORWARD: when the caller flags an emergency the controller
   stops behaving like a comfort speed tracker and demands the deceleration that
   removes the whole remaining speed error within ``emergency_stop_time_s``,
   saturated at ``brake_authority_mps2``.  Without this the controller only ever
   saw the planner's rate-limited target, produced ~0.06 m/s^2 of demand during a
   full AEB event, and the safety arbiter was the only thing in the system that
   actually braked;
5. a jerk limit on the commanded acceleration (``max_jerk_mps3``, relaxed to
   ``max_jerk_emergency_mps3`` when the caller flags an emergency).  The
   emergency jerk limit is sized from brake-system pressure build time, not from
   comfort: at the previous 15 m/s^3 the controller needed 0.53 s to reach full
   authority from coast, which is longer than the whole AEB event at urban
   speeds and was the second reason the arbiter was the only effective brake;
6. pedal rate limits (``throttle_rate_per_s``, ``brake_apply_rate_per_s``), with
   throttle release and emergency brake application exempt -- reducing tractive
   effort and applying the brake in an emergency are never rate-limited downward.

All five pieces of state are committed only after the resulting command has passed
:meth:`PIDLikeLongitudinalController._validate_command`, so a command the actuator
never received cannot become the baseline the next frame is rate-limited against.
This is defence in depth rather than a fix for an observed defect: today
``validate_motion_plan`` rejects every known raising input before any state moves.
``tests/test_control.py`` pins the property.

Guarantees (asserted in ``tests/test_control.py``)
--------------------------------------------------
* ``0 <= throttle <= max_throttle`` and ``0 <= brake <= max_brake`` always.
* ``throttle > 0`` and ``brake > 0`` never hold at the same time.
* ``throttle == 0`` on every frame the caller flags as an emergency.
* an emergency stop is held to standstill: while ``emergency`` is set and the
  vehicle is still moving toward a zero target the comfort deadband never
  releases the brake.
* ``|throttle[k] - throttle[k-1]| <= throttle_rate_per_s * dt`` for increases.
* ``|brake[k] - brake[k-1]| <= brake_apply_rate_per_s * dt`` for increases outside
  an emergency.
* the implied acceleration never changes faster than the jerk limit.
* a plan whose target speed falls at rate ``r`` produces at least ``r`` of
  commanded deceleration once the jerk limit has been served, so the vehicle
  follows the planner's ramp instead of trailing it.

Failure behaviour
-----------------
Any invalid input (non-finite or negative speed, non-finite plan, a plan that fails
:func:`adas.core.validation.validate_motion_plan`) raises
:class:`adas.core.exceptions.ControlError`.  The controller does NOT invent a
command in that case -- the caller (the safety arbiter) owns the fail-safe.

The controller is NOT thread-safe and must be one instance per pipeline.
It does not rate-limit steering: slew-rate limiting is enforced authoritatively by
:class:`adas.control.arbiter.SafetyArbiter` against the last actuated command.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from adas.core.exceptions import ControlError, ValidationError
from adas.core.logger import setup_logger
from adas.core.models import ControlCommand, MotionPlan
from adas.core.validation import validate_motion_plan

logger = setup_logger(__name__)

NOMINAL_DT_S = 0.05


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


@dataclass
class _LongitudinalState:
    """One frame of longitudinal controller output plus the state it implies.

    Returned by :meth:`PIDLikeLongitudinalController._longitudinal` and committed
    by :meth:`PIDLikeLongitudinalController.to_command` only after validation.
    """

    throttle: float
    brake: float
    accel_mps2: float
    integral_mps: float
    mode: str
    target_mps: float
    target_hist: list
    clock_s: float


@dataclass
class PIDLikeLongitudinalController:
    """Stateful PI speed controller with deadband, anti-windup, jerk and rate limits.

    The class name is retained for API compatibility with ``adas.cli`` and the
    existing tests; it is a PI controller with feed-forward, not a PID.
    """

    # --- gains and authority -------------------------------------------------
    kp_speed: float = 0.15
    """Proportional gain, (m/s^2) per (m/s) = 1/s.

    NOTE the unit change from the previous stateless controller, where this gain
    produced a pedal fraction directly.  With the default
    ``accel_authority_mps2 = 2.5`` the equivalent old pedal gain is
    ``kp_speed / 2.5``.
    """
    ki_speed: float = 0.05
    """Integral gain, (m/s^2) per (m/s . s) = 1/s^2. Removes the steady-state
    offset the old pure-P law left on any grade."""
    max_throttle: float = 1.0
    max_brake: float = 1.0
    accel_authority_mps2: float = 2.5
    brake_authority_mps2: float = 8.0

    # --- steering ------------------------------------------------------------
    max_steering_angle_deg: float = 25.0
    steering_deadband_deg: float = 0.5

    # --- shaping -------------------------------------------------------------
    speed_deadband_mps: float = 0.3
    """No pedal action while |speed error| is inside this band."""
    speed_hysteresis_mps: float = 0.15
    """Extra error required to LEAVE the deadband once inside it, and to switch
    between the throttle and brake actuators. Prevents actuator chatter."""
    emergency_stop_time_s: float = 1.0
    """Emergency feed-forward horizon, seconds.

    While ``emergency=True`` the demanded deceleration is at least
    ``|speed_error| / emergency_stop_time_s``, saturated at
    ``brake_authority_mps2``.  With the defaults (1.0 s, 8 m/s^2) any speed error
    of 8 m/s or more commands full brake, subject only to the emergency jerk
    limit.  The comfort PI law still applies and the MORE severe of the two wins.
    """
    max_jerk_mps3: float = 4.0
    """Comfort jerk limit, m/s^3."""
    max_jerk_emergency_mps3: float = 40.0
    """Emergency jerk limit, m/s^3.

    Sized from the brake actuator, not from comfort: a hydraulic service brake
    develops full deceleration in roughly 0.2 s, so 8 m/s^2 / 0.2 s = 40 m/s^3.
    At the previous 15 m/s^3 the controller took 0.53 s to reach full authority
    from coast, during which an ego at 15 m/s travels a further 6 m; the AEB
    stage was declared and actuated too slowly to matter and only the arbiter
    (which is not rate limited at all) avoided the collision.
    """
    target_ff_window: int = 21
    """Samples of target history the feed-forward slope is fitted over.

    A one-frame difference of the target is unusable as a feed-forward: a 0.2 m/s
    dither on the target is 4 m/s^2 of demand at dt = 50 ms, which alternates the
    actuators every frame (``tests/test_control.py::
    test_no_actuator_chatter_around_zero_speed_error`` fails outright with a plain
    difference).  A least-squares slope over N samples multiplies white target
    noise by ``sqrt(12 / (N (N^2-1))) / dt`` while returning a genuine ramp
    EXACTLY, at the cost of ``(N-1)/2`` frames of LAG -- 0.5 s at 20 Hz, and
    proportionally more if the pipeline runs slower, which is the one property to
    re-check if the frame rate is ever lowered.

    N and the deadband were chosen from a measured grid, not guessed
    (window 7..31 x deadband 0.1..1.0, scored on pedal frames leaked from pure
    target dither over 120,000 frames and on the min gap left in the
    "lead brakes at 4 m/s^2 from 30 m" closed loop):
    N = 11 / 0.5 leaked 16 frames in 400; N = 15 / 0.5 leaked 0 but left only
    5.4 m; N = 21 / 0.4 leaks 0 in 120,000 and leaves 7.2 m.  The window must
    also be FULL before the term is used at all -- a partial window has up to
    30x the slope variance and leaks the same dither.  Must be >= 3.
    """
    target_ff_deadband_mps2: float = 0.4
    """Soft deadband subtracted from |feed-forward|, m/s^2.

    Applied as ``sign(ff) * max(0, |ff| - deadband)``, so it is continuous -- a
    hard deadband would just move the chatter to its own edge.  With N = 21 the
    worst fitted slope over 200,000 full windows of the worst target dither in
    the test suite (uniform +/-0.2 m/s per frame, which the planner's own rate
    limiter cannot even produce: it caps the step at ``max_decel_mps2 * dt`` =
    0.15 m/s) is ~0.36 m/s^2, against an effective threshold of
    ``deadband + kp_speed * speed_deadband_mps`` = 0.445 -- a 1.24x margin, and
    zero leaked pedal frames in 120,000 measured.  The PI law covers the
    magnitude the deadband removes; it is only the standing ramp LAG that the PI
    law cannot cover, and that is what this term exists for.
    """
    target_ff_step_mps: float = 1.0
    """A target change larger than this in one frame is a STEP, not a ramp.

    Only the AEB stage can move the planner's target by this much in one frame
    (the comfort law is rate-limited to ``max_decel_mps2 * dt``).  On a step the
    regression window is discarded rather than smeared across the following
    ``target_ff_window`` frames, and that one frame uses the clamped instant
    difference; the jerk limit bounds what a single frame can do.
    """
    throttle_rate_per_s: float = 2.0
    brake_apply_rate_per_s: float = 5.0
    brake_release_rate_per_s: float = 8.0
    integral_limit_mps2: float = 1.0
    """Hard clamp on the integrator's contribution, in m/s^2."""

    # --- state (not constructor arguments) -----------------------------------
    _integral_mps: float = field(default=0.0, init=False, repr=False, compare=False)
    _prev_accel_mps2: float = field(default=0.0, init=False, repr=False, compare=False)
    _prev_throttle: float = field(default=0.0, init=False, repr=False, compare=False)
    _prev_brake: float = field(default=0.0, init=False, repr=False, compare=False)
    _mode: str = field(default="coast", init=False, repr=False, compare=False)
    _prev_target_mps: float | None = field(default=None, init=False, repr=False, compare=False)
    _target_hist: list = field(default_factory=list, init=False, repr=False, compare=False)
    """Rolling ``[(elapsed_s, target_mps), ...]`` window for the feed-forward fit."""
    _clock_s: float = field(default=0.0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.kp_speed <= 0:
            raise ValidationError(f"Speed gain must be positive, got {self.kp_speed}")
        if self.ki_speed < 0:
            raise ValidationError(f"Integral gain must be non-negative, got {self.ki_speed}")
        if self.max_steering_angle_deg <= 0:
            raise ValidationError(
                f"Max steering must be positive, got {self.max_steering_angle_deg}"
            )
        if self.accel_authority_mps2 <= 0 or self.brake_authority_mps2 <= 0:
            raise ValidationError("Actuator authorities must be positive m/s^2 values")
        if self.max_jerk_mps3 <= 0 or self.max_jerk_emergency_mps3 < self.max_jerk_mps3:
            raise ValidationError("require 0 < max_jerk_mps3 <= max_jerk_emergency_mps3")
        if not (math.isfinite(self.target_ff_deadband_mps2) and self.target_ff_deadband_mps2 >= 0):
            raise ValidationError(
                "target_ff_deadband_mps2 must be finite and non-negative, got "
                f"{self.target_ff_deadband_mps2}"
            )
        if self.target_ff_window < 3:
            raise ValidationError(
                f"target_ff_window must be >= 3, got {self.target_ff_window}"
            )
        if not (math.isfinite(self.target_ff_step_mps) and self.target_ff_step_mps > 0):
            raise ValidationError(
                f"target_ff_step_mps must be positive, got {self.target_ff_step_mps}"
            )
        if self.throttle_rate_per_s <= 0 or self.brake_apply_rate_per_s <= 0:
            raise ValidationError("Pedal rate limits must be positive")
        if self.speed_deadband_mps < 0 or self.speed_hysteresis_mps < 0:
            raise ValidationError("Deadband and hysteresis must be non-negative")
        if not (math.isfinite(self.emergency_stop_time_s) and self.emergency_stop_time_s > 0):
            raise ValidationError(
                f"emergency_stop_time_s must be positive, got {self.emergency_stop_time_s}"
            )

        logger.info(
            "Longitudinal controller initialised: kp=%.3f 1/s, ki=%.3f 1/s^2, "
            "authority +%.1f/-%.1f m/s^2, jerk<=%.1f m/s^3, max_steering=%.1f deg",
            self.kp_speed,
            self.ki_speed,
            self.accel_authority_mps2,
            self.brake_authority_mps2,
            self.max_jerk_mps3,
            self.max_steering_angle_deg,
        )

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Clear the integrator and the rate-limiter history."""
        self._integral_mps = 0.0
        self._prev_accel_mps2 = 0.0
        self._prev_throttle = 0.0
        self._prev_brake = 0.0
        self._mode = "coast"
        self._prev_target_mps = None
        self._target_hist = []
        self._clock_s = 0.0

    @property
    def commanded_accel_mps2(self) -> float:
        """Last acceleration the controller asked for, m/s^2 (negative = braking)."""
        return self._prev_accel_mps2

    # -------------------------------------------------------------- main path

    def to_command(
        self,
        plan: MotionPlan,
        current_speed_mps: float,
        dt_s: float = NOMINAL_DT_S,
        emergency: bool = False,
    ) -> ControlCommand:
        """Convert a motion plan into an actuator command.

        Args:
            plan: Motion plan with a target speed and a steering setpoint.
            current_speed_mps: Measured ego speed, m/s. Must be finite and >= 0.
            dt_s: Measured time since the previous call, seconds. Used by the jerk
                and rate limits; an unusable value falls back to 20 Hz nominal.
            emergency: True when the caller has an active hazard. Relaxes the jerk
                limit and exempts brake application from its rate limit.

        Returns:
            A :class:`ControlCommand` satisfying every guarantee in the module
            docstring.

        Raises:
            ControlError: on any invalid input. The caller owns the fail-safe.
        """
        try:
            validate_motion_plan(plan)

            if not math.isfinite(current_speed_mps):
                raise ValidationError(f"Current speed must be finite, got {current_speed_mps}")
            if current_speed_mps < 0:
                raise ValidationError(
                    f"Current speed must be non-negative, got {current_speed_mps}"
                )

            dt = dt_s if (math.isfinite(dt_s) and dt_s > 1e-4) else NOMINAL_DT_S

            # Steering first: it is stateless and it can raise, and no controller
            # state may advance on a frame whose command never reaches an actuator.
            steering = self._steering(plan.steering_angle_deg)
            state = self._longitudinal(
                plan.target_speed_mps, current_speed_mps, dt, emergency
            )

            cmd = ControlCommand(throttle=state.throttle, brake=state.brake, steering=steering)
            self._validate_command(cmd)
            if emergency and cmd.throttle > 0.0:  # pragma: no cover - defensive
                raise ValidationError("throttle commanded during an emergency")

            # Commit only now that the command is known good.
            self._integral_mps = state.integral_mps
            self._prev_accel_mps2 = state.accel_mps2
            self._prev_throttle = state.throttle
            self._prev_brake = state.brake
            self._mode = state.mode
            self._prev_target_mps = state.target_mps
            self._target_hist = state.target_hist
            self._clock_s = state.clock_s

            logger.debug(
                "Control: t=%.3f b=%.3f s=%.3f (v*=%.2f, v=%.2f, a*=%.2f m/s^2, mode=%s)",
                cmd.throttle,
                cmd.brake,
                cmd.steering,
                plan.target_speed_mps,
                current_speed_mps,
                self._prev_accel_mps2,
                self._mode,
            )
            return cmd

        except ValidationError as e:
            raise ControlError(f"Control validation failed: {e}") from e
        except ControlError:
            raise
        except Exception as e:  # pragma: no cover - defensive
            raise ControlError(f"Control command generation failed: {e}") from e

    # --------------------------------------------------------------- internals

    def _longitudinal(
        self, target_speed_mps: float, current_speed_mps: float, dt_s: float, emergency: bool
    ) -> "_LongitudinalState":
        """PI -> target-rate FF -> emergency FF -> jerk limit -> pedal map -> rate limit.

        PURE with respect to ``self``: the new controller state is returned and the
        caller commits it only once the resulting command has validated.
        """
        error = target_speed_mps - current_speed_mps

        # 1. Deadband with hysteresis on the actuator SELECTION.
        enter = self.speed_deadband_mps
        leave = self.speed_deadband_mps + self.speed_hysteresis_mps
        # Wider band to LEAVE coast than to enter it, so the two actuators cannot
        # alternate frame-to-frame around zero error.
        threshold = leave if self._mode == "coast" else enter
        in_deadband = abs(error) <= threshold

        upper = self.accel_authority_mps2 * self.max_throttle
        lower = -self.brake_authority_mps2 * self.max_brake

        # 2. PI with clamping anti-windup.
        if in_deadband:
            # Bleed the integrator rather than freezing it forever at an offset.
            integral = self._integral_mps * 0.9
            accel_cmd = 0.0
        else:
            candidate_integral = self._integral_mps + error * dt_s
            candidate_accel = self.kp_speed * error + self.ki_speed * candidate_integral
            saturated = candidate_accel > upper or candidate_accel < lower
            integral_contribution = self.ki_speed * candidate_integral
            windup = abs(integral_contribution) > self.integral_limit_mps2
            # An update that shrinks |integral| is always accepted. This is
            # defence in depth: with these limits no input sequence was found
            # that latches the plain clamping scheme, but freezing the
            # integrator in BOTH directions is the general failure mode, and
            # tests/test_control.py pins the unwind property either way.
            unwinding = abs(candidate_integral) < abs(self._integral_mps)
            if unwinding or (not saturated and not windup):
                integral = candidate_integral
            else:
                integral = self._integral_mps
            accel_cmd = self.kp_speed * error + self.ki_speed * integral
            accel_cmd = _clamp(accel_cmd, lower, upper)

        # 3. Target-rate feed-forward: dv*/dt.
        #
        #    The planner publishes a RATE-LIMITED target, so "brake at 3 m/s^2"
        #    reaches the controller as a target that falls 0.15 m/s per frame,
        #    never as a large speed error. A pure error law tracks a ramp with a
        #    standing lag of (ramp rate / kp) = 20 m/s at kp = 0.15 1/s, which is
        #    unreachable, so the controller settled at ~0.4 m/s^2 of demand while
        #    the planner was asking for 3.0 and the ego kept closing. Adding the
        #    derivative of the target removes the lag by construction and leaves
        #    the PI to trim the residual.
        #
        #    Bounds: on any frame the planner produced, |dv*/dt| is already
        #    limited to max_accel_mps2 / max_decel_mps2 by the planner's own rate
        #    limiter, so this term cannot exceed the deceleration that was asked
        #    for. The clamp to [lower, upper] bounds it for hand-built plans too,
        #    and the jerk limit below shapes the onset either way.
        target_rate, hist, clock = self._target_slope(target_speed_mps, dt_s)
        accel_cmd = _clamp(accel_cmd + _clamp(target_rate, lower, upper), lower, upper)

        # 4. Emergency feed-forward. In an emergency the plan is "stop", not "track
        #    a comfort speed profile": demand the deceleration that removes the
        #    remaining speed error inside emergency_stop_time_s and take whichever
        #    of that and the PI output is more severe. The comfort deadband does
        #    not apply -- there is no comfort requirement during an AEB event.
        if emergency and error < 0.0:
            feed_forward = max(lower, error / self.emergency_stop_time_s)
            accel_cmd = min(accel_cmd, feed_forward)

        # 5. Jerk limit on the commanded acceleration.
        jerk_limit = self.max_jerk_emergency_mps3 if emergency else self.max_jerk_mps3
        max_step = jerk_limit * dt_s
        accel_cmd = _clamp(
            accel_cmd, self._prev_accel_mps2 - max_step, self._prev_accel_mps2 + max_step
        )

        # 6. Pedal map with an actuator-switch hysteresis band.
        #
        #    The band exists to stop the two actuators hunting around zero error;
        #    during an emergency there is no comfort requirement and no throttle
        #    to hunt with, and the band actively hurts: at the end of an AEB stop
        #    the emergency demand decays as |v| / emergency_stop_time_s, so below
        #    kp_speed * speed_deadband_mps * emergency_stop_time_s (0.045 m/s
        #    with the defaults) it fell inside the band, the brake was released
        #    and the ego CREPT into the obstacle at 4 cm/s -- observed over the
        #    last 0.11 m of the 25 m stationary-lead scenario. An AEB stop must
        #    be held to standstill.
        accel_deadband = 0.0 if emergency else self.kp_speed * self.speed_deadband_mps
        if accel_cmd > accel_deadband and not emergency:
            raw_throttle = _clamp(accel_cmd / self.accel_authority_mps2, 0.0, self.max_throttle)
            raw_brake = 0.0
            mode = "throttle"
        elif accel_cmd < -accel_deadband:
            raw_throttle = 0.0
            raw_brake = _clamp(-accel_cmd / self.brake_authority_mps2, 0.0, self.max_brake)
            mode = "brake"
        else:
            raw_throttle = 0.0
            raw_brake = 0.0
            mode = "coast"

        # 7. Pedal rate limits. Throttle release and emergency braking are exempt.
        throttle = min(raw_throttle, self._prev_throttle + self.throttle_rate_per_s * dt_s)
        throttle = _clamp(throttle, 0.0, self.max_throttle)

        if emergency:
            brake = raw_brake
        else:
            brake = min(raw_brake, self._prev_brake + self.brake_apply_rate_per_s * dt_s)
        brake = max(brake, self._prev_brake - self.brake_release_rate_per_s * dt_s)
        brake = _clamp(brake, 0.0, self.max_brake)

        if brake > 0.0 or emergency:
            throttle = 0.0

        return _LongitudinalState(
            throttle=throttle,
            brake=brake,
            accel_mps2=accel_cmd,
            integral_mps=integral,
            mode=mode,
            target_mps=target_speed_mps,
            target_hist=hist,
            clock_s=clock,
        )

    def _target_slope(self, target_speed_mps: float, dt_s: float) -> tuple:
        """Least-squares dv*/dt over the rolling target window.

        PURE with respect to ``self``: returns ``(slope_mps2, new_history,
        new_clock)`` and the caller commits the history only once the resulting
        command has validated, exactly like the other four pieces of state.

        Returns 0.0 until a FULL window exists, and discards the window on a
        target STEP so that an AEB target of 0 m/s is not smeared over the
        following ``target_ff_window`` frames once the emergency clears.
        """
        clock = self._clock_s + dt_s
        hist = list(self._target_hist)

        step = (
            self._prev_target_mps is not None
            and abs(target_speed_mps - self._prev_target_mps) > self.target_ff_step_mps
        )
        if step:
            instant = (target_speed_mps - self._prev_target_mps) / dt_s
            return instant, [(clock, target_speed_mps)], clock

        hist.append((clock, target_speed_mps))
        if len(hist) > self.target_ff_window:
            del hist[0 : len(hist) - self.target_ff_window]
        # Rebase the window's clock on its oldest sample so that neither the
        # stored times nor `clock` grow without bound over a multi-hour run and
        # cost the regression its floating-point resolution.
        base = hist[0][0]
        if base != 0.0:
            hist = [(t - base, y) for t, y in hist]
            clock -= base
        if len(hist) < self.target_ff_window:
            # A PARTIAL window is not a cheap approximation of a full one: the
            # slope variance goes as 1 / (n (n^2 - 1)), so a 3-sample window is
            # 12x noisier than a 15-sample one and leaks target dither straight
            # onto the pedals (frame 2 of
            # test_feed_forward_ignores_target_noise, brake 0.024 out of nothing).
            # The feed-forward stays off until the evidence is there; the PI law
            # carries those frames exactly as it did before this term existed.
            return 0.0, hist, clock

        n = float(len(hist))
        t_mean = sum(t for t, _ in hist) / n
        y_mean = sum(y for _, y in hist) / n
        s_tt = sum((t - t_mean) ** 2 for t, _ in hist)
        if s_tt <= 1e-12:  # pragma: no cover - defensive, dt is validated > 1e-4
            return 0.0, hist, clock
        slope = sum((t - t_mean) * (y - y_mean) for t, y in hist) / s_tt
        if not math.isfinite(slope):  # pragma: no cover - defensive
            return 0.0, hist, clock

        # Soft (continuous) deadband: shrink toward zero rather than truncate, so
        # there is no edge for the actuators to chatter across.
        band = self.target_ff_deadband_mps2
        if slope > band:
            return slope - band, hist, clock
        if slope < -band:
            return slope + band, hist, clock
        return 0.0, hist, clock

    def _steering(self, target_steering_deg: float) -> float:
        """Normalise a road-wheel angle to [-1, 1] with a deadband."""
        if not math.isfinite(target_steering_deg):
            raise ValidationError(f"Invalid steering angle: {target_steering_deg}")
        if abs(target_steering_deg) < self.steering_deadband_deg:
            return 0.0
        return _clamp(target_steering_deg / self.max_steering_angle_deg, -1.0, 1.0)

    def _validate_command(self, cmd: ControlCommand) -> None:
        """Full command validation.

        ``adas.core.validation.validate_control_command`` does not check ``brake``
        at all and permits a negative throttle, so this controller checks its own
        output rather than relying on it. See the handoff note for
        ``core/validation.py``.
        """
        for name in ("throttle", "brake", "steering"):
            value = getattr(cmd, name)
            if not math.isfinite(value):
                raise ValidationError(f"{name} is not finite: {value!r}")
        if not (0.0 <= cmd.throttle <= self.max_throttle):
            raise ValidationError(f"throttle out of range: {cmd.throttle}")
        if not (0.0 <= cmd.brake <= self.max_brake):
            raise ValidationError(f"brake out of range: {cmd.brake}")
        if not (-1.0 <= cmd.steering <= 1.0):
            raise ValidationError(f"steering out of range: {cmd.steering}")
        if cmd.throttle > 0.0 and cmd.brake > 0.0:
            raise ValidationError(
                f"throttle ({cmd.throttle}) and brake ({cmd.brake}) both engaged"
            )
