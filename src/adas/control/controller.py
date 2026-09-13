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
was legal.  This version keeps four pieces of state -- the speed-error integral,
the previous commanded acceleration, the previous throttle and the previous brake --
and uses them to enforce, in order:

1. a symmetric speed-error deadband with hysteresis, so throttle and brake cannot
   alternate frame to frame around zero error;
2. a PI law with clamping anti-windup (the integrator is frozen whenever the
   acceleration command is saturated, or while inside the deadband -- except that
   an update which moves the integrator back TOWARD zero is always accepted, so a
   frozen integrator can never latch);
3. an EMERGENCY FEED-FORWARD: when the caller flags an emergency the controller
   stops behaving like a comfort speed tracker and demands the deceleration that
   removes the whole remaining speed error within ``emergency_stop_time_s``,
   saturated at ``brake_authority_mps2``.  Without this the controller only ever
   saw the planner's rate-limited target, produced ~0.06 m/s^2 of demand during a
   full AEB event, and the safety arbiter was the only thing in the system that
   actually braked;
4. a jerk limit on the commanded acceleration (``max_jerk_mps3``, relaxed to
   ``max_jerk_emergency_mps3`` when the caller flags an emergency);
5. pedal rate limits (``throttle_rate_per_s``, ``brake_apply_rate_per_s``), with
   throttle release and emergency brake application exempt -- reducing tractive
   effort and applying the brake in an emergency are never rate-limited downward.

All four pieces of state are committed only after the resulting command has passed
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
* ``|throttle[k] - throttle[k-1]| <= throttle_rate_per_s * dt`` for increases.
* ``|brake[k] - brake[k-1]| <= brake_apply_rate_per_s * dt`` for increases outside
  an emergency.
* the implied acceleration never changes faster than the jerk limit.

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
    max_jerk_emergency_mps3: float = 15.0
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
        """PI -> emergency feed-forward -> jerk limit -> pedal map -> rate limit.

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

        # 3. Emergency feed-forward. In an emergency the plan is "stop", not "track
        #    a comfort speed profile": demand the deceleration that removes the
        #    remaining speed error inside emergency_stop_time_s and take whichever
        #    of that and the PI output is more severe. The comfort deadband does
        #    not apply -- there is no comfort requirement during an AEB event.
        if emergency and error < 0.0:
            feed_forward = max(lower, error / self.emergency_stop_time_s)
            accel_cmd = min(accel_cmd, feed_forward)

        # 4. Jerk limit on the commanded acceleration.
        jerk_limit = self.max_jerk_emergency_mps3 if emergency else self.max_jerk_mps3
        max_step = jerk_limit * dt_s
        accel_cmd = _clamp(
            accel_cmd, self._prev_accel_mps2 - max_step, self._prev_accel_mps2 + max_step
        )

        # 5. Pedal map with an actuator-switch hysteresis band.
        accel_deadband = self.kp_speed * self.speed_deadband_mps
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

        # 6. Pedal rate limits. Throttle release and emergency braking are exempt.
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
        )

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
