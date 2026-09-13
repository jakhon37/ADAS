"""Behavior planning: composes the longitudinal and lateral laws into a MotionPlan.

This module is deliberately thin.  All of the decision maths lives in
:mod:`adas.planning.longitudinal` and :mod:`adas.planning.lateral`, which are pure
enough to sweep in property tests.  What this class adds is:

* lead selection (which tracked object the longitudinal law reacts to),
* the sign conversion between the tracker's ``velocity_mps`` (positive = closing)
  and the planner's ``range_rate_mps`` (``v_lead - v_ego``, negative = closing),
* an explicit ``perception_valid`` argument so that "the detector threw" can never
  be confused with "the road is empty",
* the assembly of the human-readable ``MotionPlan.reason``.

Units are as declared on each argument: metres, m/s, m/s^2, seconds, degrees of
road-wheel angle, pixels for image-space quantities.

Failure behaviour
-----------------
* ``perception_valid=False`` -- the target speed is held and ramped down; after a
  few consecutive dropouts it ramps at the controlled-stop rate.  The planner never
  returns cruise speed on a perception failure.
* Ego speed unknown -- the constant-time-gap law is undefined, so the planner
  degrades to hold-and-ramp and says so in ``reason``.  It never guesses a speed.
  **The pipeline must pass ``ego``**; see the module note in
  :meth:`BehaviorPlanner.plan`.
* Any internal error -- wrapped in :class:`adas.core.exceptions.PlanningError`.  The
  caller must treat that as a planning dropout and fail safe; it is not a
  "keep the last plan" condition.

Logging
-------
Every condition here that can persist for the whole run (no ego bus, a track with
an unusable range) is a LATCHED condition, not an event, so its WARNING goes
through a :class:`adas.planning.longitudinal.LogGate`: once on entry, at most once
per gate period while it holds, once on recovery.  The per-frame truth lives in
``MotionPlan.reason`` and in :attr:`BehaviorPlanner.ego_speed_available`, which is
what a health endpoint should read.  Before this gate existed the planner wrote one
WARNING per frame for the whole run -- 72,000 lines an hour at 20 Hz.

The planner is STATEFUL (it rate-limits the target speed against the previous one).
Call :meth:`reset` on replay restart.  It is not thread-safe; one instance per
pipeline, matching the tracker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from adas.core.exceptions import PlanningError, ValidationError
from adas.core.logger import setup_logger
from adas.core.models import EgoState, LaneModel, MotionPlan, RangeSource, TrackedObject
from adas.core.validation import validate_motion_plan
from adas.planning.lateral import CameraGeometry, LateralLimits, LateralPlanner, SteeringDecision
from adas.planning.longitudinal import (
    NOMINAL_DT_S,
    LeadVehicle,
    LogGate,
    LongitudinalLimits,
    LongitudinalPlanner,
    SpeedDecision,
)

logger = setup_logger(__name__)


@dataclass
class BehaviorPlanner:
    """Lane keeping + adaptive cruise control.

    The field names that existed before this rewrite are preserved so that
    ``adas.cli.build_pipeline`` keeps working unchanged; the new fields all have
    defaults and are documented in the handoff notes for ``core/config.py``.
    """

    # --- pre-existing, wired from PlannerConfig -----------------------------
    cruise_speed_mps: float = 15.0
    min_follow_distance_m: float = 12.0
    """``d0`` of the spacing policy, metres. See LongitudinalLimits."""
    max_steering_deg: float = 22.0
    time_gap_s: float = 2.0
    max_decel_mps2: float = 3.0
    lane_center_gain: float = 1.0
    ego_lane_half_width_frac: float = 0.0
    """Fraction of the frame width, either side of the lane centre, that counts as
    the ego lane.  0 disables gating -- which means braking for oncoming and parked
    traffic, so a non-zero value should be configured for any real deployment."""

    # --- new, not yet in PlannerConfig (see handoff) -------------------------
    max_accel_mps2: float = 2.0
    standstill_gap_m: float = 4.0
    k_distance: float = 0.4
    k_speed: float = 0.6
    aeb_ttc_s: float = 0.9
    warn_ttc_s: float = 1.6
    emergency_decel_mps2: float = 8.0
    mrm_decel_mps2: float = 3.5
    wheelbase_m: float = 2.8
    max_lateral_accel_mps2: float = 3.0
    steering_speed_ref_mps: float = 12.0
    camera: CameraGeometry | None = None
    """Camera extrinsics. None (the default) forces the non-metric steering law."""

    degraded_log_period_s: float = 60.0
    """Seconds between repeats of a WARNING that describes a latched condition."""

    longitudinal: LongitudinalPlanner = field(init=False, repr=False, compare=False)
    lateral: LateralPlanner = field(init=False, repr=False, compare=False)
    last_speed_decision: SpeedDecision | None = field(default=None, init=False, repr=False, compare=False)
    last_steering_decision: SteeringDecision | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.cruise_speed_mps <= 0:
            raise ValidationError(f"Cruise speed must be positive, got {self.cruise_speed_mps}")
        if self.min_follow_distance_m < 0:
            raise ValidationError(
                f"Min follow distance must be non-negative, got {self.min_follow_distance_m}"
            )
        if self.max_steering_deg <= 0:
            raise ValidationError(f"Max steering must be positive, got {self.max_steering_deg}")

        self._ego_invalid_gate = LogGate(self.degraded_log_period_s)
        self._bad_range_gate = LogGate(self.degraded_log_period_s)

        self.longitudinal = LongitudinalPlanner(
            LongitudinalLimits(
                cruise_speed_mps=self.cruise_speed_mps,
                min_follow_distance_m=self.min_follow_distance_m,
                time_gap_s=self.time_gap_s,
                standstill_gap_m=min(self.standstill_gap_m, self.min_follow_distance_m),
                k_distance=self.k_distance,
                k_speed=self.k_speed,
                max_accel_mps2=self.max_accel_mps2,
                max_decel_mps2=self.max_decel_mps2,
                emergency_decel_mps2=max(self.emergency_decel_mps2, self.max_decel_mps2),
                mrm_decel_mps2=self.mrm_decel_mps2,
                aeb_ttc_s=self.aeb_ttc_s,
                warn_ttc_s=max(self.warn_ttc_s, self.aeb_ttc_s),
            ),
            ego_speed_log_period_s=self.degraded_log_period_s,
        )
        self.lateral = LateralPlanner(
            LateralLimits(
                max_steering_deg=self.max_steering_deg,
                wheelbase_m=self.wheelbase_m,
                max_lateral_accel_mps2=self.max_lateral_accel_mps2,
                lane_center_gain=self.lane_center_gain,
                speed_ref_mps=self.steering_speed_ref_mps,
            ),
            camera=self.camera,
            degraded_log_period_s=self.degraded_log_period_s,
        )

        logger.info(
            "BehaviorPlanner initialised: cruise=%.1f m/s, d0=%.1f m, T=%.1f s, "
            "a_max=%.1f/%.1f m/s^2, a_lat_max=%.1f m/s^2, metric_steering=%s",
            self.cruise_speed_mps,
            self.min_follow_distance_m,
            self.time_gap_s,
            self.max_accel_mps2,
            self.max_decel_mps2,
            self.max_lateral_accel_mps2,
            bool(self.camera is not None and self.camera.is_configured()),
        )

    # ------------------------------------------------------------------ api

    def reset(self) -> None:
        """Clear the rate limiter, dropout counters and log latches.

        Call on replay restart.
        """
        self.longitudinal.reset()
        self.lateral.reset()
        self._ego_invalid_gate.reset()
        self._bad_range_gate.reset()
        self.last_speed_decision = None
        self.last_steering_decision = None

    @property
    def ego_speed_available(self) -> bool:
        """False while the longitudinal law is latched in ``ego_speed_unavailable``.

        The machine-readable form of the condition that used to be a per-frame
        WARNING.  ``MotionPlan.reason`` carries the same fact on every frame.
        """
        return self.longitudinal.ego_speed_available

    def plan(
        self,
        frame_width_px: int,
        lane_center_px: float | None,
        objects: list[TrackedObject],
        ego: EgoState | None = None,
        ego_speed_mps: float | None = None,
        perception_valid: bool = True,
        dt_s: float = NOMINAL_DT_S,
        lane: LaneModel | None = None,
        frame_height_px: int | None = None,
    ) -> MotionPlan:
        """Generate one motion plan.

        Args:
            frame_width_px: Image width in pixels. Must be > 0.
            lane_center_px: Lane-centre column, or None when the lane is unknown.
            objects: Tracked objects from this frame. An EMPTY LIST MEANS THE ROAD
                IS EMPTY. If perception failed, pass ``perception_valid=False``
                instead of an empty list.
            ego: Measured ego state. ``ego.valid`` must be True for
                ``ego.speed_mps`` to be used.
            ego_speed_mps: Alternative to ``ego`` for callers that only have a
                speed. Ignored when ``ego`` is valid.
            perception_valid: False when the detector or lane estimator failed.
            dt_s: Measured time since the previous call, seconds.
            lane: Full lane model, when available (used for the confidence gate).
            frame_height_px: Image height; needed for the metric steering law.

        Returns:
            A validated :class:`MotionPlan`.

        Raises:
            PlanningError: on any internal failure. Callers must fail safe.

        Note:
            If neither ``ego`` nor ``ego_speed_mps`` is supplied the longitudinal
            law degrades to hold-and-ramp-down, because a constant-time-gap policy
            without ego speed is not defined. Supplying ego speed is mandatory for
            normal operation.
        """
        try:
            if frame_width_px <= 0:
                raise ValidationError(f"Invalid frame width: {frame_width_px}")

            speed = self._resolve_ego_speed(ego, ego_speed_mps)
            lead = self._select_lead(frame_width_px, lane_center_px, objects)

            speed_decision = self.longitudinal.plan(
                lead=lead,
                ego_speed_mps=speed,
                perception_valid=perception_valid,
                dt_s=dt_s,
            )
            steering_decision = self.lateral.plan(
                frame_width_px=frame_width_px,
                lane_center_px=None if not perception_valid else lane_center_px,
                ego_speed_mps=speed,
                lane=lane,
                frame_height_px=frame_height_px,
            )

            self.last_speed_decision = speed_decision
            self.last_steering_decision = steering_decision

            speed_reason = speed_decision.reason
            if speed_decision.degraded:
                # Still honest about what is happening: degraded, and whether there
                # is a lead we are decelerating behind.
                prefix = "follow_degraded" if lead is not None else "degraded"
                speed_reason = "%s_%s" % (prefix, speed_decision.reason)

            plan = MotionPlan(
                target_speed_mps=speed_decision.target_speed_mps,
                steering_angle_deg=steering_decision.steering_angle_deg,
                reason="%s|%s" % (speed_reason, steering_decision.reason),
            )
            validate_motion_plan(plan)

            logger.debug(
                "Plan: v*=%.2f m/s (%s, ttc=%.2f s, a_req=%.2f m/s^2), delta=%.2f deg (%s)",
                plan.target_speed_mps,
                speed_reason,
                speed_decision.ttc_s,
                speed_decision.required_decel_mps2,
                plan.steering_angle_deg,
                steering_decision.reason,
            )
            return plan

        except ValidationError as e:
            raise PlanningError(f"Planning validation failed: {e}") from e
        except PlanningError:
            raise
        except Exception as e:  # pragma: no cover - defensive
            raise PlanningError(f"Planning failed: {e}") from e

    # --------------------------------------------------------------- helpers

    def _resolve_ego_speed(
        self, ego: EgoState | None, ego_speed_mps: float | None
    ) -> float | None:
        """Pick the ego speed to plan with, or None when there is no usable one.

        A bus that reports ``valid=True`` with a nonsense speed is a latched fault
        on a broken vehicle, so the WARNING is gated rather than emitted per frame.
        """
        if ego is not None and ego.valid:
            if math.isfinite(ego.speed_mps) and ego.speed_mps >= 0.0:
                recovered, dropped = self._ego_invalid_gate.clear()
                if recovered:
                    logger.info(
                        "EgoState speed usable again (%.2f m/s) after %d suppressed "
                        "warnings",
                        ego.speed_mps,
                        dropped,
                    )
                return ego.speed_mps
            emit, dropped = self._ego_invalid_gate.mark()
            if emit:
                logger.warning(
                    "EgoState marked valid but speed is %r; ignoring. Condition is "
                    "latched: repeats at most every %.0f s (%d frames suppressed "
                    "since the last line).",
                    ego.speed_mps,
                    self._ego_invalid_gate.period_s,
                    dropped,
                )
        if ego_speed_mps is not None and math.isfinite(ego_speed_mps) and ego_speed_mps >= 0.0:
            return ego_speed_mps
        return None

    def _objects_ahead(
        self,
        frame_width_px: int,
        lane_center_px: float | None,
        objects: list[TrackedObject],
    ) -> list[TrackedObject]:
        """Keep only the objects the ego is going to drive into.

        Uses ``TrackedObject.in_ego_lane`` when the perception stack populates it,
        otherwise falls back to an image-x band of half width
        ``ego_lane_half_width_frac * frame_width_px`` around the lane centre.
        """
        if any(obj.in_ego_lane for obj in objects):
            return [obj for obj in objects if obj.in_ego_lane]
        frac = self.ego_lane_half_width_frac
        if frac <= 0.0 or frame_width_px <= 0:
            return list(objects)
        cx_ref = lane_center_px if lane_center_px is not None else frame_width_px / 2.0
        half = frame_width_px * frac
        return [
            obj
            for obj in objects
            if abs((obj.box.x1 + obj.box.x2) / 2.0 - cx_ref) <= half
        ]

    def _select_lead(
        self,
        frame_width_px: int,
        lane_center_px: float | None,
        objects: list[TrackedObject],
    ) -> LeadVehicle | None:
        """Nearest plausible in-path object, converted into a :class:`LeadVehicle`.

        Objects with a non-finite range are dropped with a warning rather than
        silently treated as infinitely far away.
        """
        best: TrackedObject | None = None
        excluded: list[int] = []
        for obj in self._objects_ahead(frame_width_px, lane_center_px, objects):
            if not math.isfinite(obj.distance_m) or obj.distance_m < 0.0:
                excluded.append(obj.track_id)
                continue
            if best is None or obj.distance_m < best.distance_m:
                best = obj
        if excluded:
            # A range estimator that is broken stays broken; gate the line so a
            # permanently bad track cannot write one WARNING per frame per track.
            emit, dropped = self._bad_range_gate.mark()
            if emit:
                logger.warning(
                    "Tracks %s have an implausible range; excluded from lead "
                    "selection. Condition is latched: repeats at most every %.0f s "
                    "(%d frames suppressed since the last line).",
                    excluded,
                    self._bad_range_gate.period_s,
                    dropped,
                )
        else:
            self._bad_range_gate.clear()
        if best is None:
            return None

        distance_m = best.distance_m
        source = RangeSource.PINHOLE
        confidence = best.box.confidence
        if best.range_estimate is not None and best.range_estimate.source != RangeSource.UNAVAILABLE:
            estimate = best.range_estimate
            if math.isfinite(estimate.distance_m) and estimate.distance_m >= 0.0:
                # Take the conservative (nearer) of the two channels; the arbiter
                # separately flags a disagreement.
                distance_m = min(distance_m, estimate.distance_m)
                source = estimate.source
                confidence = min(confidence, estimate.confidence) if estimate.confidence > 0 else confidence

        # TrackedObject.velocity_mps is positive-when-closing; the planner wants
        # v_rel = v_lead - v_ego, which is negative-when-closing.
        rate = -best.velocity_mps if math.isfinite(best.velocity_mps) else 0.0
        return LeadVehicle(
            distance_m=distance_m,
            range_rate_mps=rate,
            track_id=best.track_id,
            confidence=confidence,
            frames_since_measurement=max(0, best.time_since_update),
            source=source,
        )
