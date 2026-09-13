"""Safety monitor: the front door to the authoritative safety arbiter.

The class :class:`SafetyMonitor` exists for two reasons:

1. It owns :meth:`SafetyMonitor.arbitrate`, which is the ONLY authoritative safety
   entry point in the system.  It returns an
   :class:`adas.core.models.ArbitrationResult` whose ``command`` is what the
   actuators must receive.  All of the reasoning lives in
   :class:`adas.control.arbiter.SafetyArbiter`; this class just holds one and maps
   :class:`SafetyLimits` onto :class:`adas.control.arbiter.ArbiterLimits`.
2. It keeps the previous raise-based methods alive so that callers that have not
   yet migrated (``adas.runtime.pipeline``) keep working.  Those methods are
   ADVISORY: they detect and report, they do not prevent.  Every one of them says
   so in its own docstring.  Nothing safety-relevant may depend on them.

What changed in the advisory API, and why
-----------------------------------------
* ``check_motion_plan`` no longer raises on a large requested DECELERATION.  The
  old code vetoed exactly the correct response to danger (ego at 20 m/s, stopped
  car at 3 m: the only complaint the monitor made was about the braking) while
  merely logging excessive acceleration.  Excessive requested acceleration -- the
  direction that adds energy -- is now the branch that raises.
* ``check_control_command`` used to compute a steering rate into a local variable
  and discard it.  It now really checks the rate, on the PHYSICAL road-wheel angle
  and against a REAL elapsed time, and returns a rate-limited command.  It still
  does not raise on a rate violation, because the pipeline calls it outside its
  own error handling and an exception there would drop the frame entirely -- which
  is a worse failure than a logged, corrected command.  Authoritative enforcement
  is in :meth:`arbitrate`.
* ``sanitize_control_command`` no longer turns a NaN brake into full braking.
  ``max(0.0, min(1.0, nan))`` is ``1.0`` in Python, so the old clamp silently
  converted a corrupt command into an ABS-limit stop.  A non-finite field now
  produces an all-zero command and an ERROR log; the arbiter escalates to a
  controlled stop when it sees the same input.
* Negative throttle is clamped to 0, not to -1.  ``core/validation.py`` still
  permits throttle in [-1, 1]; see the handoff note.

Units: metres, m/s, m/s^2, radians, seconds, and dimensionless actuator commands.
Stateful and NOT thread-safe; one instance per pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from adas.control.arbiter import ArbiterLimits, SafetyArbiter, SafetyContext
from adas.core.exceptions import SafetyViolation
from adas.core.logger import setup_logger
from adas.core.models import (
    ArbitrationResult,
    ControlCommand,
    MotionPlan,
    SafetyState,
    TrackedObject,
)

logger = setup_logger(__name__)


@dataclass
class SafetyLimits:
    """Safety limits for ADAS operation.

    The first eight fields are the ones ``adas.cli.build_pipeline`` wires from
    ``SafetyConfig``; the rest default to values that are stricter than or equal to
    the planner's.

    :meth:`to_arbiter_limits` projects EVERY field the arbiter reads.  What is NOT
    yet true: ``adas.cli.build_safety_limits`` does not have a ``SafetyConfig`` key
    for the fields below the "previously unreachable from configuration" marker, so
    those are reachable programmatically but not yet from a YAML file.  That file is
    not owned by this module; see the handoff note.

    ``max_lateral_offset_m`` is the one conditional case -- it can only be enforced
    when a CALIBRATED lane geometry supplies a metric
    ``SafetyContext.lateral_offset_m``; without one the arbiter reports
    ``lane_offset_unavailable`` instead of silently passing.
    """

    max_speed_mps: float = 33.0
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 8.0
    max_steering_rate_rad_s: float = 0.5
    max_steering_angle_rad: float = 0.52
    min_following_distance_m: float = 2.0
    """ABSOLUTE floor on the gap, metres. This is not a following policy -- at
    33 m/s it is a 0.06 s gap. The real headway rule is the RSS test inside the
    arbiter, parameterised by ``reaction_time_s`` and the two braking
    capabilities."""
    max_lateral_offset_m: float = 1.5
    """Lane-departure limit in metres. See the class docstring: enforced only when
    a metric lateral offset is available."""
    plan_horizon_s: float = 1.0
    """Horizon used to judge a plan's requested ACCELERATION only."""

    # --- new; see the handoff note for SafetyConfig --------------------------
    max_jerk_mps3: float = 4.0
    max_jerk_emergency_mps3: float = 15.0
    max_lateral_accel_mps2: float = 4.5
    wheelbase_m: float = 2.8
    max_road_wheel_rad: float = 0.436
    """Road-wheel angle at ``steering = 1.0``. Must equal
    ``radians(ControllerConfig.max_steering_angle_deg)``."""
    brake_authority_mps2: float = 8.0
    accel_authority_mps2: float = 2.5
    standstill_gap_m: float = 4.0
    reaction_time_s: float = 0.6
    ego_brake_capability_mps2: float = 6.0
    lead_brake_capability_mps2: float = 8.0
    ttc_brake_s: float = 0.9
    ttc_warn_s: float = 1.6
    comfort_decel_mps2: float = 3.0
    mrm_decel_mps2: float = 3.5
    limited_after_dropouts: int = 1
    mrm_after_dropouts: int = 3
    disengage_after_frames: int = 40
    recovery_frames: int = 10

    # --- previously unreachable from configuration ---------------------------
    # Every one of these existed only as an ArbiterLimits default, so a threshold
    # that can latch the TERMINAL state (range_disagreement_frac, min_dt_s) could
    # not be changed for a deployment without editing source. They are defaulted
    # to the ArbiterLimits values, so projecting them changes no behaviour.
    range_disagreement_frac: float = 0.30
    min_range_confidence: float = 0.35
    range_corroboration_frames: int = 3
    min_dt_s: float = 0.005
    max_dt_s: float = 0.5
    max_frame_gap_s: float = 0.5
    aeb_required_decel_mps2: float = 5.0
    aeb_decel_margin: float = 1.15
    aeb_min_decel_mps2: float = 4.0
    aeb_headway_frac: float = 0.5
    kinematics_min_speed_mps: float = 0.5
    limited_throttle_cap: float = 0.0
    throttle_rate_per_s: float = 5.0
    brake_release_rate_per_s: float = 8.0
    min_in_path_half_width_frac: float = 0.30
    """Floor on the arbiter's in-path corridor half width, as a fraction of the
    frame width. Keep it >= the planner's ``ego_lane_half_width_frac``: the
    backstop must never look at a narrower slice of the road than the planner."""
    lane_trust_confidence: float = 0.50
    mrm_straighten_speed_mps: float = 1.0
    allow_uncalibrated_range: bool = False
    """Opt-in switch. Leave False and an uncalibrated camera reports
    ``camera_uncalibrated`` and floors the state at LIMITED."""

    # --- evidence required before a closing RATE may authorise full braking ---
    deferred_aeb_decel_mps2: float = 5.0
    """Deceleration ceiling while an ACUTE emergency test has tripped on the safe
    prior but the closing rate is not yet measured. Must sit between
    ``comfort_decel_mps2`` and ``max_deceleration_mps2``."""
    aeb_rate_corroboration_frames: int = 2
    """Consecutive frames a rate-dependent emergency test must hold before it
    authorises full-authority braking."""
    aeb_min_rate_samples: int = 4
    """Raw range measurements the arbiter needs before it calls its closing rate
    MEASURED. Below this the rate is still the ``-ego_speed`` safe prior, and the
    rate-dependent emergency tests are held to the graded response."""
    aeb_min_rate_span_s: float = 0.15
    """Elapsed time the same window must span."""
    range_rate_window_s: float = 0.60
    """Window the measured range slope is fitted over."""

    # --- range-source stability ------------------------------------------------
    range_source_dwell_frames: int = 5
    range_confidence_hysteresis: float = 0.10
    range_disagreement_hysteresis: float = 0.25

    max_coast_frames: int = 5
    """Frames of tracker coasting the arbiter will keep assessing a lead for during
    a perception dropout, instead of forgetting the hazard it was braking for."""

    def to_arbiter_limits(self) -> ArbiterLimits:
        """Project these limits onto the arbiter's own limit object.

        EVERY :class:`~adas.control.arbiter.ArbiterLimits` field that the arbiter
        reads is set here. The previous version silently dropped
        ``range_disagreement_frac``, ``min_dt_s``, ``max_dt_s``,
        ``max_frame_gap_s``, the whole ``aeb_*`` group,
        ``kinematics_min_speed_mps``, ``limited_throttle_cap`` and the output
        shaping rates, so those thresholds were unreachable from ``SafetyConfig``
        and from the YAML -- including the two that could latch the terminal
        DISENGAGE state.

        Note: ``adas.cli.build_safety_limits`` still has to grow the matching
        ``SafetyConfig`` keys before the new fields are reachable from a config
        FILE; that file is not owned by this module. Until then they are
        reachable programmatically and default to the arbiter's own values.
        """
        return ArbiterLimits(
            max_speed_mps=self.max_speed_mps,
            max_acceleration_mps2=self.max_acceleration_mps2,
            max_deceleration_mps2=self.max_deceleration_mps2,
            comfort_decel_mps2=self.comfort_decel_mps2,
            mrm_decel_mps2=self.mrm_decel_mps2,
            max_jerk_mps3=self.max_jerk_mps3,
            max_jerk_emergency_mps3=self.max_jerk_emergency_mps3,
            accel_authority_mps2=self.accel_authority_mps2,
            brake_authority_mps2=self.brake_authority_mps2,
            max_steering_angle_rad=self.max_steering_angle_rad,
            max_steering_rate_rad_s=self.max_steering_rate_rad_s,
            max_lateral_accel_mps2=self.max_lateral_accel_mps2,
            max_lateral_offset_m=self.max_lateral_offset_m,
            wheelbase_m=self.wheelbase_m,
            max_road_wheel_rad=self.max_road_wheel_rad,
            absolute_min_gap_m=self.min_following_distance_m,
            standstill_gap_m=max(self.standstill_gap_m, self.min_following_distance_m),
            reaction_time_s=self.reaction_time_s,
            ego_brake_capability_mps2=self.ego_brake_capability_mps2,
            lead_brake_capability_mps2=self.lead_brake_capability_mps2,
            ttc_brake_s=self.ttc_brake_s,
            ttc_warn_s=max(self.ttc_warn_s, self.ttc_brake_s),
            plan_horizon_s=self.plan_horizon_s,
            limited_after_dropouts=self.limited_after_dropouts,
            mrm_after_dropouts=self.mrm_after_dropouts,
            disengage_after_frames=self.disengage_after_frames,
            recovery_frames=self.recovery_frames,
            limited_throttle_cap=self.limited_throttle_cap,
            kinematics_min_speed_mps=self.kinematics_min_speed_mps,
            aeb_required_decel_mps2=self.aeb_required_decel_mps2,
            aeb_decel_margin=self.aeb_decel_margin,
            aeb_min_decel_mps2=self.aeb_min_decel_mps2,
            aeb_headway_frac=self.aeb_headway_frac,
            min_dt_s=self.min_dt_s,
            max_dt_s=self.max_dt_s,
            max_frame_gap_s=self.max_frame_gap_s,
            range_disagreement_frac=self.range_disagreement_frac,
            min_range_confidence=self.min_range_confidence,
            range_corroboration_frames=self.range_corroboration_frames,
            min_in_path_half_width_frac=self.min_in_path_half_width_frac,
            lane_trust_confidence=self.lane_trust_confidence,
            mrm_straighten_speed_mps=self.mrm_straighten_speed_mps,
            allow_uncalibrated_range=self.allow_uncalibrated_range,
            throttle_rate_per_s=self.throttle_rate_per_s,
            brake_release_rate_per_s=self.brake_release_rate_per_s,
            deferred_aeb_decel_mps2=self.deferred_aeb_decel_mps2,
            aeb_rate_corroboration_frames=self.aeb_rate_corroboration_frames,
            aeb_min_rate_samples=self.aeb_min_rate_samples,
            aeb_min_rate_span_s=self.aeb_min_rate_span_s,
            range_rate_window_s=max(self.range_rate_window_s, self.aeb_min_rate_span_s),
            range_source_dwell_frames=self.range_source_dwell_frames,
            range_confidence_hysteresis=self.range_confidence_hysteresis,
            range_disagreement_hysteresis=self.range_disagreement_hysteresis,
            max_coast_frames=self.max_coast_frames,
        )


class SafetyMonitor:
    """Holds the arbiter and the (advisory) legacy checks."""

    def __init__(
        self,
        limits: SafetyLimits | None = None,
        arbiter: SafetyArbiter | None = None,
    ) -> None:
        """Build a monitor.

        Args:
            limits: Safety limits. Defaults are used when None.
            arbiter: An explicit arbiter, mainly for tests. When None one is built
                from ``limits``.
        """
        self.limits = limits or SafetyLimits()
        self.arbiter = arbiter or SafetyArbiter(self.limits.to_arbiter_limits())
        self.advisory_violations = 0
        """Count of advisory checks that reported a problem. Diagnostics only --
        the number of times the ARBITER intervened is ``arbiter.total_overrides``."""
        self._last_steering_rad = 0.0
        self._last_speed_mps: float | None = None
        self._last_accel_mps2: float | None = None

    # --------------------------------------------------------- authoritative

    def arbitrate(
        self,
        plan: MotionPlan | None,
        command: ControlCommand,
        state: SafetyContext,
    ) -> ArbitrationResult:
        """Authoritative arbitration. ``result.command`` is what the actuators get.

        See :meth:`adas.control.arbiter.SafetyArbiter.arbitrate`. The caller MUST
        NOT actuate ``command`` when the result disagrees with it, and MUST NOT
        swallow the result's state: a state other than
        :attr:`~adas.core.models.SafetyState.NOMINAL` is a real degradation that
        has to reach the driver interface.
        """
        return self.arbiter.arbitrate(plan, command, state)

    @property
    def state(self) -> SafetyState:
        """The arbiter's latched safety state."""
        return self.arbiter.state

    def reset(self) -> None:
        """Clear every latch, counter and history, including a DISENGAGE latch."""
        self.arbiter.reset()
        self.advisory_violations = 0
        self._last_steering_rad = 0.0
        self._last_speed_mps = None
        self._last_accel_mps2 = None

    # ---------------------------------------------------------- advisory API

    def check_motion_plan(self, plan: MotionPlan, current_speed_mps: float) -> None:
        """ADVISORY. Raise if the plan is outside the envelope.

        Detects, does not prevent. The pipeline that calls this must migrate to
        :meth:`arbitrate`; until it does, a violation here is logged and counted
        and the plan is still executed.

        Args:
            plan: Proposed motion plan.
            current_speed_mps: Measured ego speed, m/s.

        Raises:
            SafetyViolation: target speed above ``max_speed_mps``, steering angle
                above ``max_steering_angle_rad``, or a requested ACCELERATION above
                ``max_acceleration_mps2`` over ``plan_horizon_s``. A requested
                deceleration is never a violation here.
        """
        if not math.isfinite(plan.target_speed_mps):
            self.advisory_violations += 1
            raise SafetyViolation(f"Target speed is not finite: {plan.target_speed_mps!r}")

        if plan.target_speed_mps > self.limits.max_speed_mps:
            self.advisory_violations += 1
            raise SafetyViolation(
                f"Target speed {plan.target_speed_mps:.1f} m/s exceeds limit "
                f"{self.limits.max_speed_mps:.1f} m/s"
            )

        if not math.isfinite(plan.steering_angle_deg):
            self.advisory_violations += 1
            raise SafetyViolation(f"Steering angle is not finite: {plan.steering_angle_deg!r}")

        steering_rad = math.radians(plan.steering_angle_deg)
        if abs(steering_rad) > self.limits.max_steering_angle_rad:
            self.advisory_violations += 1
            raise SafetyViolation(
                f"Steering angle {steering_rad:.3f} rad exceeds limit "
                f"±{self.limits.max_steering_angle_rad:.3f} rad"
            )

        if math.isfinite(current_speed_mps):
            horizon = max(self.limits.plan_horizon_s, 1e-3)
            requested_accel = (plan.target_speed_mps - current_speed_mps) / horizon
            if requested_accel > self.limits.max_acceleration_mps2:
                self.advisory_violations += 1
                raise SafetyViolation(
                    f"Requested acceleration {requested_accel:.2f} m/s² exceeds limit "
                    f"{self.limits.max_acceleration_mps2:.2f} m/s² over a "
                    f"{horizon:.2f} s horizon"
                )
            # Deceleration is deliberately NOT a violation here. Whether a large
            # deceleration is justified depends on the hazard, which this method
            # is not given; the arbiter judges the ACHIEVED deceleration with that
            # context.
            if -requested_accel > self.limits.max_deceleration_mps2:
                logger.debug(
                    "Plan requests %.2f m/s² of deceleration over %.2f s; "
                    "justification is judged by the arbiter, not here",
                    -requested_accel,
                    horizon,
                )

    def check_control_command(
        self, cmd: ControlCommand, dt: float = 0.05
    ) -> ControlCommand:
        """ADVISORY. Check and rate-limit the steering of a command.

        Converts the normalised steering to a physical road-wheel angle using
        ``max_road_wheel_rad`` and enforces ``max_steering_rate_rad_s`` against the
        supplied elapsed time.  Returns a corrected command and logs; it does not
        raise on a rate violation, because the current pipeline calls this outside
        its own error handling and dropping the frame would be worse.

        Args:
            cmd: The command about to be actuated.
            dt: MEASURED elapsed time since the previous command, seconds.

        Returns:
            The command with its steering slew-rate limited.

        Raises:
            SafetyViolation: only when a field is not finite, which is not
                recoverable by clamping.
        """
        for name in ("throttle", "brake", "steering"):
            value = getattr(cmd, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                self.advisory_violations += 1
                raise SafetyViolation(f"Control command field {name} is not finite: {value!r}")

        step = dt if (math.isfinite(dt) and dt > 1e-4) else 0.05
        angle_rad = max(-1.0, min(1.0, cmd.steering)) * self.limits.max_road_wheel_rad
        max_step = self.limits.max_steering_rate_rad_s * step
        delta = angle_rad - self._last_steering_rad
        if abs(delta) > max_step + 1e-9:
            self.advisory_violations += 1
            logger.warning(
                "Steering rate %.3f rad/s exceeds limit %.3f rad/s; limiting %.3f -> %.3f rad",
                abs(delta) / step,
                self.limits.max_steering_rate_rad_s,
                angle_rad,
                self._last_steering_rad + math.copysign(max_step, delta),
            )
            angle_rad = self._last_steering_rad + math.copysign(max_step, delta)

        self._last_steering_rad = angle_rad
        return ControlCommand(
            throttle=cmd.throttle,
            brake=cmd.brake,
            steering=max(-1.0, min(1.0, angle_rad / self.limits.max_road_wheel_rad)),
        )

    def check_following_distance(
        self, lead_vehicle: TrackedObject | None, ego_speed_mps: float
    ) -> None:
        """ADVISORY. Raise when the gap is below the RSS minimum for this speed.

        Unlike the previous version this is speed AND lead-speed dependent: it
        computes the RSS minimum gap

            ``d_min = v_ego*t_react + v_ego²/(2*a_ego) - v_lead²/(2*a_lead)``

        floored at ``min_following_distance_m``.  The old code raised only below a
        fixed 2.0 m (0.06 s of headway at 33 m/s) and downgraded the useful,
        speed-dependent test to a log line.

        Note:
            ``TrackedObject.velocity_mps`` is positive-when-closing, so the lead
            speed is ``ego_speed - velocity_mps``.

        Args:
            lead_vehicle: Lead object, or None.
            ego_speed_mps: Measured ego speed, m/s.

        Raises:
            SafetyViolation: when the gap is below the RSS minimum.
        """
        if lead_vehicle is None:
            return
        distance_m = lead_vehicle.distance_m
        if not math.isfinite(distance_m):
            self.advisory_violations += 1
            raise SafetyViolation(f"Lead vehicle range is not finite: {distance_m!r}")
        if not math.isfinite(ego_speed_mps) or ego_speed_mps < 0.0:
            ego_speed_mps = 0.0

        closing = lead_vehicle.velocity_mps if math.isfinite(lead_vehicle.velocity_mps) else 0.0
        lead_speed_mps = max(0.0, ego_speed_mps - closing)
        minimum = (
            ego_speed_mps * self.limits.reaction_time_s
            + (ego_speed_mps ** 2) / (2.0 * self.limits.ego_brake_capability_mps2)
            - (lead_speed_mps ** 2) / (2.0 * self.limits.lead_brake_capability_mps2)
        )
        minimum = max(self.limits.min_following_distance_m, minimum)

        if distance_m < minimum:
            self.advisory_violations += 1
            raise SafetyViolation(
                f"Following distance {distance_m:.1f}m below minimum {minimum:.1f}m "
                f"(ego {ego_speed_mps:.1f} m/s, lead {lead_speed_mps:.1f} m/s)"
            )
        if distance_m < 1.3 * minimum:
            logger.warning(
                "Following distance %.1fm within 30%% of the RSS minimum %.1fm",
                distance_m,
                minimum,
            )

    def sanitize_control_command(self, cmd: ControlCommand) -> ControlCommand:
        """Clamp a command into the valid actuator ranges.

        A non-finite field yields an all-zero command and an ERROR log rather than
        a clamp: ``max(0.0, min(1.0, nan))`` is ``1.0`` in Python, so clamping a
        NaN brake would silently command an ABS-limit stop.

        Negative throttle is clamped to 0.0. Simultaneous throttle and brake is
        resolved in favour of the brake.

        Args:
            cmd: Input command.

        Returns:
            A command with ``throttle`` and ``brake`` in [0, 1] and ``steering`` in
            [-1, 1].
        """
        values = (cmd.throttle, cmd.brake, cmd.steering)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            self.advisory_violations += 1
            logger.error(
                "Non-finite control command %r; substituting a neutral command. The "
                "arbiter escalates this to a controlled stop.",
                cmd,
            )
            return ControlCommand(throttle=0.0, brake=0.0, steering=0.0)

        throttle = max(0.0, min(1.0, float(cmd.throttle)))
        brake = max(0.0, min(1.0, float(cmd.brake)))
        steering = max(-1.0, min(1.0, float(cmd.steering)))
        if brake > 0.0 and throttle > 0.0:
            self.advisory_violations += 1
            logger.warning(
                "Throttle %.2f and brake %.2f both engaged; dropping throttle",
                throttle,
                brake,
            )
            throttle = 0.0

        if throttle != cmd.throttle or brake != cmd.brake or steering != cmd.steering:
            logger.warning(
                "Control command clamped: throttle %.2f->%.2f, brake %.2f->%.2f, "
                "steering %.3f->%.3f",
                cmd.throttle,
                throttle,
                cmd.brake,
                brake,
                cmd.steering,
                steering,
            )
        return ControlCommand(throttle=throttle, brake=brake, steering=steering)
