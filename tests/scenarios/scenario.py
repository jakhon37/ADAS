"""Declarative scenarios, the closed loop that runs them, and the judgement.

A :class:`Scenario` says what the world does and what the system is required to
do about it.  Crucially it also carries ``physics``: a written argument, printed
by the report, for why the expected outcome is the correct one.  An expectation
without that argument is a guess, and guesses are what this harness exists to
replace.

Two classes of assertion carry equal weight here:

* **No collision when one was avoidable.**  A system that will not intervene is
  not a safety system.
* **No intervention when none was warranted.**  A system that brakes at random
  is not a safety system either; it is a new hazard.  Full-authority braking on
  a motorway for a lead that never moved is not a lesser failure than a missed
  brake, and a harness that only measures the first is how this codebase
  oscillated between the two.

Plus one that would have caught all three of the self-sustaining feedback loops
in the history of this module:

* **Recovery.**  Once the hazard is over, the system must return to NOMINAL
  within a bounded number of frames.

And four that the first version of this file left entirely unmeasured, each of
which let a real, measured defect through:

* **Bounded jerk.**  The demanded deceleration may not change faster than an
  occupant can brace for.  The shipped library measured 156.7 m/s^3 across a
  single 50 ms frame and nothing failed.
* **Never both pedals.**  Throttle and brake are never commanded together.  The
  plant used to net them into ``2.5*throttle - 8.0*brake`` and the conflict
  vanished into the arithmetic.
* **The system's own findings count.**  ``ArbitrationResult.violations`` was
  recorded, printed, and asserted on by nothing, so the arbiter could report
  ``jerk_32.2_above_15.0`` about its own command and pass.  A self-report that
  is a statement about the system's OWN output, or about the motion the system
  actually produced, is now a finding here too -- see
  :data:`ASSERTABLE_SELF_REPORTS`.
* **The sub-emergency band.**  The phantom test asked for
  ``commanded >= 3.5 m/s^2 or state is MRM``, so any unjustified braking below
  AEB grade was invisible.  Proportionality is now policed over the whole range
  of the brake: see :data:`oracle.HEADWAY_DECEL_ALLOWANCE_MPS2`.

Five correctness bugs in this judgement layer, found by audit and fixed here
---------------------------------------------------------------------------
Every one of them made the harness say something untrue, and three of them said
it about the SYSTEM when the fault was in the harness:

1. **The oracle and the plant disagreed about the vehicle.**  Avoidability was
   judged against a zero-latency, step-braking car while the system under test
   has 55 ms of sense latency, a two-frame wait for a closing rate to exist at
   all, and a jerk ceiling on its own demand.  Four scenarios were placed inside
   that fictional boundary and could only be passed by braking before the
   measurement existed.  :func:`feasibility` now prints the AFFORDABLE DECISION
   LATENCY for every scenario, and a negative one is reported as a defect in the
   scenario (``infeasible_clearance``, ``unactionable_scenario``,
   ``collided_unavoidable``) rather than in the system.
2. **The plant did not stop at contact**, so a collided run carried on and the
   report quoted a minimum gap measured from inside the other vehicle.
3. **``phantom_intervention`` fired on frames with no braking**, because
   "emergency authority" was ``decel >= 3.5 OR state is MRM``.  A state label
   with no actuation is not an intervention; it is now
   ``unwarranted_authority_state``, a separate finding with a separate severity.
4. **``excess_jerk`` punished early intervention and brake release.**  It now
   assesses only a RISING demand and takes its band from the demand rather than
   from whether the oracle's causal requirement had already become visible.
5. **``forbid_emergency_intervention`` had no read site.**  It was set on eight
   scenarios and asserted nothing.  :func:`unread_expectation_fields` now makes
   that class of defect a test failure.

And one device that would have caught all of them at once:
:class:`ReferenceController`, a constant time gap plus a braking law derived
from required deceleration, driven through the same sensor and the same plant.
Every scenario must be passable by it; ``report.py --reference`` says whether
they are.

Finally, the stack is DEGRADABLE.  The arbiter's entire design rationale is that
it is an independent backstop, and that claim is untestable while every run
executes planner, controller and arbiter together and only ever actuates the
arbiter's answer.  :class:`StackSpec` lets a scenario blind the planner, jam the
controller, or strip the arbiter of its authority, so that "the arbiter alone
can stop this vehicle" and "the primary path alone can stop this vehicle" are
two separate, separately falsifiable claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Dict, List, Optional, Sequence, Tuple

from adas.core.models import (
    ArbitrationResult,
    ControlCommand,
    EgoState,
    MotionPlan,
    PerceptionStatus,
    SafetyState,
)
from adas.control import PIDLikeLongitudinalController, SafetyMonitor
from adas.control.arbiter import SafetyContext
from adas.planning import BehaviorPlanner

from tests.scenarios import oracle as truth
from tests.scenarios.plant import (
    DEFAULT_PLANT,
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    CAMERA_FOCAL_PX,
    CAR_WIDTH_M,
    EGO_HALF_WIDTH_M,
    MAX_ROAD_WHEEL_RAD,
    PERFECT_PERCEPTION,
    WHEELBASE_M,
    LeadSpec,
    Observation,
    PerceptionSpec,
    Plant,
    PlantConfig,
    RoadSpec,
    Sensor,
    WorldState,
    lead_constant_speed,
    lead_stationary,
    straight_road,
)

PLANNER_EGO_LANE_HALF_WIDTH_FRAC = 0.25
"""Ego-lane half width used by the planner, as a fraction of the frame width.

0.25 of 1280 px is 320 px either side of the lane centre.  At the 20 m lookahead
this harness renders boxes for, 320 px is 7 m, comfortably wider than a lane and
narrow enough to exclude the next lane but one.  The planner's shipped default
is 0.0 (no gating at all), which would brake for parked and oncoming traffic; no
deployment would run that, so the harness does not test it.
"""

ARBITER_EGO_LANE_HALF_WIDTH_FRAC = 0.30
"""What the pipeline passes the arbiter.  Wider than the planner's, so the
arbiter's hazard window is never narrower than the planner's."""

STANDSTILL_MPS = 0.5
"""Below this true speed the ego counts as stopped.

A stopped vehicle holding its brake is not intervening in anything, so those
frames are exempt from the phantom and justification tests.
"""

PEDAL_CONFLICT_EPS = 0.01
"""Smallest simultaneous throttle and brake demand that counts as a conflict.

One per cent of pedal travel is below the resolution of any real actuator and
below the quantisation of a CAN torque request, so anything smaller is
arithmetic noise rather than a command.  Anything larger is the system asking
the vehicle to accelerate and to decelerate at the same time, and there is no
manoeuvre for which that is the answer: whichever pedal wins, the other one was
wrong, and on a real vehicle the pair is a brake-drag fault that overheats the
discs and desensitises the driver's own pedal.
"""

ASSERTABLE_SELF_REPORTS = (
    "command_throttle_brake_conflict",
    "command_not_finite",
    "command_throttle_out_of_range",
    "command_brake_out_of_range",
    "command_steering_out_of_range",
    "plan_target_speed_invalid",
    "plan_steering_invalid",
    "decision_path_exception",
    "jerk_",
    "measured_accel_",
    "measured_decel_",
)
"""Prefixes of ``ArbitrationResult.violations`` that this harness asserts on.

The rule that decides membership, stated once so that the list can be extended
without re-arguing it:

    A self-report is ASSERTABLE when it is a statement about the system's own
    OUTPUT, or about the system's own PROPRIOCEPTION -- the vehicle's motion as
    the vehicle bus reports it.  It is NOT assertable when it derives from
    EXTEROCEPTION: anything the camera, the tracker or the lane estimator said.

By that rule ``jerk_32.2_above_15.0`` is assertable -- it is differentiated from
the CAN speed signal, which is the system's own sense of its own body -- and so
is ``command_throttle_brake_conflict``, which is a defect in the bytes the
primary path emitted.  ``gap_8.10m_below_absolute_min`` is not: it describes
traffic, and traffic is the oracle's job.  ``perception_dropout_7`` is not: the
scenario injected that dropout on purpose, and reporting it is the system
working.  ``camera_uncalibrated``, ``range_jump_track1`` and the ``timing_*``
family are all in the same position -- true statements about an input this
harness chose.

``lane_departure_*`` is the interesting exclusion, and it is why the line is
drawn at proprioception rather than at "motion".  The arbiter computes it from
``SafetyContext.lateral_offset_m``, which is the LANE ESTIMATOR's opinion of
where the vehicle is, and a scenario that injects ``lane_offset_error_m`` makes
the arbiter file 275 lane departures for a vehicle that never left its lane.
Believing that report would be scoring the system on how thoroughly it was
lied to.  Real lane departure is asserted from the plant's true lateral offset
instead, through :attr:`Expectation.max_abs_lateral_offset_m`, which is ground
truth and cannot be corrupted by a sensor spec.

Matching is by PREFIX, which is what disposes of the arbiter's own excuses.  The
arbiter files a jerk it caused itself as ``jerk_32.2_above_15.0_arbiter_induced``
and demotes it to advisory so that it cannot change the state.  That demotion is
sound reasoning about CONTROL -- a system must not treat its own actuator as
fresh evidence -- and it is irrelevant to ACCEPTANCE.  The occupant's neck does
not care which subsystem commanded the jerk.  The suffix is an attribution, and
an attribution is not an exemption, so ``jerk_`` matches both spellings.
"""

_NON_ASSERTABLE_NOTE = (
    "hazard, perception, timing and range findings are statements about the "
    "world or about what the harness fed the system, and are judged by the "
    "oracle rather than taken on the system's own word"
)


# --------------------------------------------------------------------------- #
# The system under test
# --------------------------------------------------------------------------- #


PLANNER_REAL = "real"
"""The shipped :class:`BehaviorPlanner`, given everything perception reported."""

PLANNER_BLIND = "blind"
"""The shipped planner, given an EMPTY object list on every frame.

Perception is healthy and the arbiter still receives the real tracks; only the
planner's view of them is gone.  This is not a contrived failure: an object list
that is dropped, filtered out by an in-path gate, or lost to a planner-side
exception all look exactly like this from the planner's output, and the whole
claim being tested is that the arbiter does not need the planner to notice a
hazard.  With this mode a scenario asks: CAN THE ARBITER ALONE STOP THE CAR?
"""

PLANNER_RUNAWAY = "runaway"
"""The shipped planner, with its target speed overridden upward every frame.

Models a planner that has decided the road is clear and wants to accelerate --
the demand the arbiter has to veto.
"""

CONTROLLER_REAL = "real"
"""The shipped :class:`PIDLikeLongitudinalController`."""

CONTROLLER_STUCK_THROTTLE = "stuck_throttle"
"""The primary path emits full throttle every frame, whatever the plan says.

A jammed pedal, a stuck output stage, a controller that has wound its integrator
into the stop.  The arbiter is the only thing between this and the obstacle.
"""

CONTROLLER_STUCK_BRAKE = "stuck_brake"
"""The primary path emits full brake every frame, whatever the plan says.

The mirror image, and the one this codebase actually shipped: a
planner-originated brake demand on a clear road.  The arbiter is meant to be an
independent authority over the longitudinal channel, so it must be able to VETO
this, not merely pass it through.
"""

ARBITER_REAL = "real"
"""The arbiter's command is what reaches the actuators.  How the pipeline runs."""

ARBITER_BYPASS = "bypass"
"""The PRIMARY PATH's command reaches the actuators; the arbiter is advisory.

The arbiter still runs and its state and violations are still recorded -- they
are wanted as evidence -- but its command is discarded.  With this mode a
scenario asks the converse question: CAN THE PRIMARY PATH ALONE STOP THE CAR?
Both answers must be yes.  A backstop that is load-bearing is not a backstop,
and a primary path that cannot stop is one fault away from a collision.
"""


@dataclass(frozen=True)
class StackSpec:
    """Which parts of the longitudinal path are real for this scenario.

    The default is the shipped configuration.  Every other combination
    deliberately removes or corrupts one layer so that the remaining layers can
    be held to the same physical requirement on their own.

    Attributes:
        planner: One of :data:`PLANNER_REAL`, :data:`PLANNER_BLIND`,
            :data:`PLANNER_RUNAWAY`.
        controller: One of :data:`CONTROLLER_REAL`,
            :data:`CONTROLLER_STUCK_THROTTLE`, :data:`CONTROLLER_STUCK_BRAKE`.
        arbiter: One of :data:`ARBITER_REAL`, :data:`ARBITER_BYPASS`.
        runaway_target_factor: Multiplier on the cruise speed used by
            :data:`PLANNER_RUNAWAY`.
    """

    planner: str = PLANNER_REAL
    controller: str = CONTROLLER_REAL
    arbiter: str = ARBITER_REAL
    runaway_target_factor: float = 1.5

    def __post_init__(self) -> None:
        if self.planner not in (PLANNER_REAL, PLANNER_BLIND, PLANNER_RUNAWAY):
            raise ValueError("unknown planner mode %r" % (self.planner,))
        if self.controller not in (
            CONTROLLER_REAL,
            CONTROLLER_STUCK_THROTTLE,
            CONTROLLER_STUCK_BRAKE,
        ):
            raise ValueError("unknown controller mode %r" % (self.controller,))
        if self.arbiter not in (ARBITER_REAL, ARBITER_BYPASS):
            raise ValueError("unknown arbiter mode %r" % (self.arbiter,))

    @property
    def label(self) -> str:
        """Compact identifier for the report and the JSON artifact."""
        return "planner=%s,controller=%s,arbiter=%s" % (
            self.planner,
            self.controller,
            self.arbiter,
        )

    @property
    def is_default(self) -> bool:
        """True for the shipped configuration."""
        return (
            self.planner == PLANNER_REAL
            and self.controller == CONTROLLER_REAL
            and self.arbiter == ARBITER_REAL
        )

    @property
    def arbiter_only(self) -> bool:
        """True when nothing but the arbiter can produce a brake."""
        return self.arbiter == ARBITER_REAL and self.planner in (
            PLANNER_BLIND,
            PLANNER_RUNAWAY,
        )


DEFAULT_STACK = StackSpec()
"""Planner, controller and arbiter all real; the arbiter's command actuated."""

ARBITER_ONLY_STACK = StackSpec(planner=PLANNER_BLIND)
"""The planner cannot see the obstacle.  Only the arbiter can stop the vehicle."""

PRIMARY_ONLY_STACK = StackSpec(arbiter=ARBITER_BYPASS)
"""The arbiter has no authority.  Only the primary path can stop the vehicle."""


@dataclass
class FrameRecord:
    """One frame of the closed loop: truth in, command out.

    Three commands are kept, because with a degradable stack they are no longer
    the same object and the difference between them IS the evidence:

    * ``raw_command`` -- what the primary path (planner + controller) asked for.
    * ``arbiter_command`` -- what the arbiter returned when handed that.
    * ``command`` -- what the plant actually received, which is the arbiter's
      answer under :data:`ARBITER_REAL` and the primary path's under
      :data:`ARBITER_BYPASS`.
    """

    frame: int
    t_s: float
    true: WorldState
    perception_ok: bool
    detected: bool
    plan_reason: str
    plan_target_mps: float
    raw_command: ControlCommand
    command: ControlCommand
    safety_state: SafetyState
    violations: List[str] = field(default_factory=list)
    arbiter_command: Optional[ControlCommand] = None

    @property
    def commanded_decel_mps2(self) -> float:
        """The deceleration the actuated command asks for, m/s^2."""
        return self.command.brake * DEFAULT_PLANT.max_brake_decel_mps2

    @property
    def raw_decel_mps2(self) -> float:
        """The deceleration the PRIMARY PATH asked for, m/s^2."""
        return self.raw_command.brake * DEFAULT_PLANT.max_brake_decel_mps2

    @property
    def arbiter_decel_mps2(self) -> float:
        """The deceleration the ARBITER asked for, m/s^2.

        Equal to :attr:`commanded_decel_mps2` unless the arbiter is bypassed.
        """
        cmd = self.arbiter_command if self.arbiter_command is not None else self.command
        return cmd.brake * DEFAULT_PLANT.max_brake_decel_mps2

    @property
    def pedal_conflict(self) -> bool:
        """The ACTUATED command asks for throttle and brake at the same time."""
        return (
            self.command.throttle > PEDAL_CONFLICT_EPS
            and self.command.brake > PEDAL_CONFLICT_EPS
        )


_ARBITER_TAKES_CAPTURE_TIME = "measurement_t_s" in {
    f.name for f in dataclass_fields(SafetyContext)
}
"""Whether the arbiter under test accepts a separate CAPTURE time.

``SafetyContext.measurement_t_s`` is the time the measurements were TRUE, which
is ``sense_latency_s`` earlier than the decision time ``t_s``.  It is what every
range fit must be placed on, ``adas.runtime.pipeline`` supplies it from
``frame.timestamp_s``, and :class:`ReferenceController` reads the same
``obs.measurement_t_s``; withholding it from the system under test handed it a
worse input than either, and once the latency exceeded one frame period that
showed up as a fabricated closing rate (measured: 9.23 m/s reported for a true
20 m/s at 80 ms of latency).

Probed rather than assumed because ``tests/test_backtest.py`` runs THIS harness
against the two historical arbiters, whose ``SafetyContext`` predates the field.
Passing an unknown keyword there is a ``TypeError`` and the backtest -- the
harness's own proof that it still catches the two known defects -- dies with it.
"""


def _CAPTURE_TIME_KWARG(measurement_t_s: float) -> Dict[str, float]:
    """``{"measurement_t_s": t}`` for an arbiter that takes it, else ``{}``."""
    if _ARBITER_TAKES_CAPTURE_TIME:
        return {"measurement_t_s": measurement_t_s}
    return {}


class StackUnderTest:
    """The production longitudinal path, wired the way the pipeline wires it.

    planner -> controller -> arbiter, with the ARBITER'S command actuated.  That
    last point is the whole reason this harness exists: the arbiter is
    authoritative, so what it returns is what the plant receives.

    The class also implements the documented fail-safe contract (ADAS-DEC-21):
    when the decision path raises, or when the run ends, the actuators receive a
    defined command derived from the arbiter with a failed perception status --
    never a latched previous command.  It is reimplemented here rather than
    borrowed from the runner because the runner's version reads
    ``time.monotonic`` and this harness must be deterministic.
    """

    def __init__(
        self, cruise_speed_mps: float, spec: StackSpec = DEFAULT_STACK
    ) -> None:
        self.spec = spec
        self.cruise_speed_mps = float(cruise_speed_mps)
        self.planner = BehaviorPlanner(
            cruise_speed_mps=cruise_speed_mps,
            ego_lane_half_width_frac=PLANNER_EGO_LANE_HALF_WIDTH_FRAC,
        )
        self.controller = PIDLikeLongitudinalController()
        self.monitor = SafetyMonitor()
        self._last_speed_mps = cruise_speed_mps

    # -- the primary path, degradable ------------------------------------- #

    def _plan(self, obs: Observation, dt_s: float) -> MotionPlan:
        """The planner's output for this frame, under :attr:`spec`.

        Under :data:`PLANNER_BLIND` the REAL planner runs against an empty
        object list.  Perception is untouched and the arbiter is still handed
        ``obs.tracks``: the hazard is in the world and in the arbiter's input,
        and only the planner has lost it.
        """
        objects = [] if self.spec.planner == PLANNER_BLIND else obs.tracks
        plan = self.planner.plan(
            frame_width_px=obs.frame_width_px,
            lane_center_px=obs.lane_center_px,
            objects=objects,
            ego=obs.ego,
            perception_valid=obs.perception.ok,
            dt_s=dt_s,
            lane=obs.lane,
            frame_height_px=obs.frame_height_px,
        )
        if self.spec.planner == PLANNER_RUNAWAY:
            return MotionPlan(
                target_speed_mps=self.cruise_speed_mps * self.spec.runaway_target_factor,
                steering_angle_deg=plan.steering_angle_deg,
                reason="runaway_planner",
            )
        return plan

    def _primary_command(
        self, plan: MotionPlan, obs: Observation, dt_s: float
    ) -> ControlCommand:
        """The primary path's actuator demand, under :attr:`spec`.

        The jammed modes bypass the controller entirely rather than feeding it a
        silly plan, because a controller with intact rate limits would smooth a
        silly plan into something reasonable and the point is to present the
        arbiter with a demand it must veto outright.
        """
        if self.spec.controller == CONTROLLER_STUCK_THROTTLE:
            return ControlCommand(1.0, 0.0, plan.steering_angle_deg * 0.0)
        if self.spec.controller == CONTROLLER_STUCK_BRAKE:
            return ControlCommand(0.0, 1.0, plan.steering_angle_deg * 0.0)
        emergency = "aeb" in (plan.reason or "") or "emergency" in (plan.reason or "")
        return self.controller.to_command(
            plan, obs.ego.speed_mps, dt_s=dt_s, emergency=emergency
        )

    def step(
        self, obs: Observation, dt_s: float
    ) -> Tuple[ControlCommand, ControlCommand, ControlCommand, object]:
        """Run one frame.

        Returns:
            ``(actuated, primary_command, arbiter_command, arbitration_result)``.
            ``actuated`` is ``arbiter_command`` under :data:`ARBITER_REAL` and
            ``primary_command`` under :data:`ARBITER_BYPASS`.
        """
        self._last_speed_mps = obs.ego.speed_mps
        try:
            plan = self._plan(obs, dt_s)
            raw = self._primary_command(plan, obs, dt_s)
        except Exception as exc:  # noqa: BLE001 - the pipeline's own fail-safe contract
            fail = self.fail_safe(obs.t_s, obs.ego.speed_mps, dt_s)
            result = ArbitrationResult(
                command=fail,
                state=self.monitor.state,
                violations=["decision_path_exception"],
                reason=type(exc).__name__,
            )
            result.plan_reason = "exception:%s" % type(exc).__name__
            result.plan_target = 0.0
            return fail, ControlCommand(0.0, 0.0, 0.0), fail, result

        ctx = SafetyContext(
            **_CAPTURE_TIME_KWARG(obs.measurement_t_s),
            ego=obs.ego,
            tracks=obs.tracks,
            perception=obs.perception,
            dt_s=dt_s,
            timestamp_s=obs.t_s,
            frame_width_px=obs.frame_width_px,
            frame_height_px=obs.frame_height_px,
            lane=obs.lane,
            ego_lane_half_width_frac=ARBITER_EGO_LANE_HALF_WIDTH_FRAC,
            lateral_offset_m=obs.lateral_offset_m,
            camera_calibrated=True,
        )
        result = self.monitor.arbitrate(plan, raw, ctx)
        result.plan_reason = plan.reason
        result.plan_target = plan.target_speed_mps
        actuated = raw if self.spec.arbiter == ARBITER_BYPASS else result.command
        return actuated, raw, result.command, result

    def fail_safe(self, t_s: float, ego_speed_mps: float, dt_s: float) -> ControlCommand:
        """The command to actuate when the decision path is gone, or at exit.

        The arbiter is run with no plan, a neutral command and a failed
        perception status.  Per its own contract that puts it into a
        minimum-risk manoeuvre and returns its rate-shaped braking command.

        With the arbiter bypassed there is no such contract to appeal to, so the
        primary path supplies its own: the controller is driven with a
        zero-target emergency plan, which is the only stop the planner and
        controller can express between them.  Borrowing the arbiter's fail-safe
        here would smuggle back exactly the authority the scenario removed.
        """
        if self.spec.arbiter == ARBITER_BYPASS:
            return self.controller.to_command(
                MotionPlan(
                    target_speed_mps=0.0, steering_angle_deg=0.0, reason="fail_safe_stop"
                ),
                ego_speed_mps,
                dt_s=dt_s,
                emergency=True,
            )
        ctx = SafetyContext(
            ego=EgoState(speed_mps=ego_speed_mps, valid=True, timestamp_s=t_s),
            tracks=[],
            perception=PerceptionStatus(ok=False, consecutive_failures=99, reason="exit"),
            dt_s=dt_s,
            timestamp_s=t_s,
            frame_width_px=FRAME_WIDTH_PX,
            frame_height_px=FRAME_HEIGHT_PX,
        )
        return self.monitor.arbitrate(None, ControlCommand(0.0, 0.0, 0.0), ctx).command

    def shutdown(self, t_s: float, dt_s: float, hold_steps: int = 4) -> List[ControlCommand]:
        """Commands written to the actuators after the frame loop ends.

        A clean exit, a SIGTERM and a source loss all leave the loop; leaving is
        not a command, and whatever was last written stays latched on the
        actuators.  The contract is that a defined command is emitted, repeatedly
        (the arbiter's brake demand is rate limited, so one command stops the
        ramp partway).
        """
        out: List[ControlCommand] = []
        t = t_s
        for _ in range(hold_steps):
            t += dt_s
            out.append(self.fail_safe(t, self._last_speed_mps, dt_s))
        return out


# --------------------------------------------------------------------------- #
# The reference controller: proof that the specification is satisfiable
# --------------------------------------------------------------------------- #


class ReferenceController:
    """A simple, honest longitudinal policy that every scenario must be passable by.

    THIS IS NOT THE REDESIGN and it does not belong in ``src/``.  It is a
    harness fixture with exactly one job: to make the harness's own
    satisfiability checkable.  A specification is only worth something if some
    correct system can meet it, and the fastest way to find out that a scenario
    demands the impossible is to run a competent, uncomplicated controller
    against it and watch it fail.  Version 2 of this library placed four
    scenarios about a metre inside a boundary that no system with the pipeline's
    own latency could reach; this class would have found that in one run.

    It also gives the redesign a KNOWN-ACHIEVABLE TARGET.  Every number the
    redesign has to beat is printed by ``report.py --reference``.

    What it is
    ----------
    Two laws and a rate limiter, and nothing else:

    * **Constant time gap.**  The policy spacing is
      ``STANDOFF_M + TIME_GAP_S * v`` -- 12 m plus two seconds, which is 52 m at
      20 m/s and is the vehicle's own following policy.  Inside it the ego
      settles at a speed deficit proportional to the shortfall and capped at
      :data:`HEADWAY_DV_MAX`, because a gap only opens while the ego is slower
      than the lead and 3 m/s of deficit opens 40 m in a comfortable thirteen
      seconds.  The authority this law may use is capped at
      :data:`oracle.HEADWAY_DECEL_ALLOWANCE_MPS2`: headway keeping is not
      collision avoidance and must never be mistaken for it.
    * **A braking law derived from required deceleration.**  Textbook, computed
      from the MEASURED range, the ego's own closing rate estimate and an
      estimate of the lead's acceleration: if the lead is decelerating it will
      still travel ``v_lead^2 / 2|a_lead|``, so the ego has that plus the gap
      less the standstill clearance in which to stop; if it is not, the ego has
      only to wash out the relative speed inside the gap.  No margin factors, no
      state machine, no thresholds tuned against a recording.
    * **A jerk limiter** at exactly the ceilings this specification derives:
      :data:`oracle.COMFORT_JERK_MPS3` while the demand stays inside the comfort
      band and :data:`oracle.EMERGENCY_JERK_MPS3` once it is heading past
      emergency grade.

    What it refuses to do, and why it matters
    -----------------------------------------
    * **It never brakes on a rate it has not measured.**  A camera measures
      range; a closing rate is a difference of ranges over time.  Until two
      distinct captures exist there is no rate, and the only prior available --
      "assume the object is stationary in the world" -- is exactly what produced
      the constant-range phantom braking.  Without a rate the controller falls
      back on the time-gap law, which is bounded at 1.5 m/s^2.
    * **It rejects a range jump.**  A reported range that moves further in one
      frame than any pair of road vehicles could move re-anchors the estimate
      and discards the rate rather than being differentiated into a 200 m/s
      closure.  :data:`RANGE_JUMP_M` is 3.0 m, which at 20 Hz is 60 m/s of
      implied closure and ten standard deviations of the harness's own range
      noise.
    * **It holds through a detection miss.**  A detector that reports nothing
      has not reported that the road is clear, so the avoidance demand is held
      for :data:`MISS_HOLD_FRAMES` before it decays.
    * **It never commands both pedals.**

    It pays the same latency the system under test pays -- it is driven from the
    same :class:`tests.scenarios.plant.Observation`, one frame stale, through the
    same plant and the same actuator lag -- so a scenario it cannot pass is a
    scenario nothing can pass.
    """

    TIME_GAP_S = 2.0
    """Following time gap, seconds.  The vehicle's own policy, and the figure the
    scenario library quotes as ``12 m + 2.0 s x v``."""

    STANDOFF_M = 12.0
    """Standing part of the policy gap, metres."""

    TARGET_CLEARANCE_M = truth.REQUIRED_CLEARANCE_M + 0.25
    """Room a completed stop aims to leave, metres.

    The oracle's 2.0 m plus a quarter of a metre of aim-off, which is the range
    the controller cannot see: it is acting on a measurement one frame old, and
    a law that aims at exactly the required clearance therefore stops a fraction
    inside it every time.  0.25 m is one frame of the closure that is still
    present in the last metre of a completed stop."""

    HEADWAY_DV_MAX = 3.0
    """Largest speed deficit the time-gap law will hold, m/s.

    A gap opens at the deficit, so 3 m/s opens 40 m -- the worst shortfall any
    scenario here presents -- in about thirteen seconds, and costs 3 m/s of
    speed.  Larger deficits open the gap faster and are what a passenger reads
    as the car giving up on the journey."""

    HEADWAY_GAIN = 0.30
    """Speed deficit per metre of gap shortfall, 1/s."""

    SPEED_GAIN = 0.8
    """Deceleration demanded per m/s of speed error, 1/s."""

    THROTTLE_GAIN = 0.35
    """Throttle fraction per m/s of speed deficit."""

    RAMP_ALLOWANCE_S = 0.4
    """Cap on the closure conceded to the brake's own build-up, seconds.

    While the demand ramps at :data:`oracle.EMERGENCY_JERK_MPS3` the average
    deceleration is about half the target, so the required-deceleration law has
    to give away roughly half the ramp time's worth of closure or it chases its
    own lag and settles high.  The allowance is computed FROM THE RAMP ACTUALLY
    NEEDED -- ``(target - current) / 20 m/s^3`` -- rather than being a fixed
    fraction of a second, because a brake that is already applied needs no
    allowance at all, and charging one anyway inflates the demand by a sixth
    exactly where the case is tightest.  0.4 s is the full-range ramp and
    therefore the ceiling."""

    DEMAND_MARGIN_FRAC = 0.05
    DEMAND_MARGIN_MPS2 = 0.05
    """Prudence added to the computed requirement.

    Five per cent and 0.05 m/s^2, which is the quantisation of a brake command
    that has been through a rate limiter.  Deliberately small: the requirement
    is recomputed every frame from a fresh measurement, so a standing margin
    buys nothing that the next frame does not buy anyway, and the oracle's
    proportionality test measures the response against the requirement rather
    than against the outcome."""

    ACCEL_CONFIDENCE_SIGMAS = 5.0
    """Confidence demanded of the LEAD ACCELERATION term, in standard errors.

    Higher than :data:`CONFIDENCE_SIGMAS` and not for statistical reasons -- the
    false-alarm rate a given sigma count buys is the same for both -- but
    because the CONSEQUENCE is not.  A 3-sigma false closure enters the braking
    law squared and small: at a 20 m gap it produces 0.15 m/s^2, which nobody
    feels.  A 3-sigma false lead deceleration switches the law to its
    lead-is-stopping branch, where the demand scales with the EGO's speed
    squared and not with the error, and produces 4.7 m/s^2 -- a full emergency
    manufactured out of range noise, which is this codebase's founding defect.
    The bound is therefore set so that the worst demand pure noise can produce
    through this term stays below the headway allowance; at the harness's own
    +/-0.30 m of range noise the split-half acceleration estimate has a standard
    error of about 15 m/s^2, so five of them is 75 m/s^2 and no draw survives it."""

    BLIND_HOLD_FRAMES = 8
    """Frames of perception loss tolerated before a minimum-risk stop begins.

    0.4 s.  A single dropped frame is a dropped frame; half a second of nothing
    is a vehicle driving blind, and 0.4 s at 20 m/s is 8 m travelled without a
    picture."""

    MRM_DECEL_MPS2 = 3.0
    """Deceleration of the minimum-risk stop, m/s^2.

    The comfort limit: the vehicle has to stop, nothing has been detected in
    front of it, and the traffic behind has no reason to expect more."""

    MISS_HOLD_FRAMES = 24
    """Frames an avoidance demand survives a detection miss.

    1.2 s.  A detector that produced nothing has not produced evidence of an
    empty road, and the harness's own dropout scenarios inject 0.8 s of it."""

    RANGE_JUMP_M = 3.0
    """Range discontinuity treated as a re-anchor rather than as motion."""

    RATE_MIN_SAMPLES = 4
    """Fewest range samples the avoidance law will act on.

    Three intervals.  Two samples give a slope with no residual, so there is no
    way to tell a rate from a noise draw; four give two degrees of freedom and
    the first honest estimate of how much of the slope is measurement error.
    Below this the controller falls back on the time-gap law, which is bounded
    at 1.5 m/s^2 and cannot hurt anyone."""

    RATE_WINDOW = 9
    """Least-squares window for the controller's own range differentiation.

    0.45 s.  The standard error of a slope fitted over ``n`` samples spaced
    ``dt`` is ``sigma / sqrt(dt^2 n (n^2-1) / 12)``, so widening the window from
    5 to 9 samples cuts the noise on the rate by a factor of 2.4 while costing
    about 0.2 s of lag against a lead whose deceleration is changing -- which
    every scenario here can afford, and none of them can afford a 6 m/s^2 brake
    for a range estimate that wandered."""

    CONFIDENCE_SIGMAS = 4.0
    """Confidence demanded before the braking law engages, in standard errors.

    THE GATE IS ON THE BOUND; THE MAGNITUDE IS THE ESTIMATE.  The controller
    will not begin braking for collision avoidance until the closure it has
    measured is four standard errors clear of zero -- the errors being computed
    from the residuals of its own fit, so a noise-free sensor is not penalised
    (the residuals are zero and the bound is the estimate) and a noisy one
    cannot manufacture a hazard.  Once the gate is open the law uses the
    UNBIASED estimate, because the required deceleration goes as the square of
    the closure and braking for a deliberately pessimistic closure would be its
    own kind of over-response.

    Four rather than three because the question is not "is this frame a false
    alarm?" but "does this controller ever brake for nothing across the whole
    corpus?".  Fifty scenarios of three hundred frames is fifteen thousand
    opportunities, and a one-sided three-sigma gate (1.3e-3) would be expected
    to open about twenty times; four sigma (3.2e-5) is expected to open once in
    two corpora.  The cost of the extra sigma is nothing when the sensor is
    clean and about one frame of latency when it is not, because a genuine
    20 m/s closure is more than seven standard errors even on the shortest
    window this controller will fit."""

    LAT_WN = 2.0
    LAT_ZETA = 0.9
    """Natural frequency and damping of the lane-keeping loop, rad/s.

    Lateral acceleration is ``v^2 (delta / L + kappa)``, so a proportional-
    derivative law placed at 2.0 rad/s leaves a steady-state offset of
    ``v^2 kappa / wn^2`` against a constant bend: 0.43 m on the 230 m radius
    this library uses and 0.25 m on the 400 m one, both inside the 1.75 m the
    scenarios allow.  The lateral half exists only so that the longitudinal
    scenarios can be run on a bend at all; it is not a proposal."""

    def __init__(self, cruise_speed_mps: float, spec: StackSpec = DEFAULT_STACK) -> None:
        self.spec = spec
        self.cruise_speed_mps = float(cruise_speed_mps)
        self._decel = 0.0
        self._blind = 0
        self._range_hist: Dict[int, List[Tuple[float, float]]] = {}
        self._vego_hist: List[Tuple[float, float]] = []
        self._sigma_range: Optional[float] = None
        self._sigma_var: Optional[float] = None
        self._miss_frames = 0
        self._held_avoid = 0.0
        self._last_offset: Optional[float] = None
        self._last_offset_t: Optional[float] = None
        self._offset_rate = 0.0
        self._steer = 0.0
        self._last_speed_mps = float(cruise_speed_mps)
        self._last_brake = 0.0

    # ------------------------------------------------------------ estimation

    SIGMA_EWMA_ALPHA = 0.1
    """Forgetting factor of the running measurement-noise estimate.

    See :meth:`_note_noise` for how the estimate is formed.

    The noise on a range measurement is a property of the SENSOR, not of the
    last four samples, so it is estimated once and remembered.  This is not a
    refinement: with a four-sample fit the residuals carry two degrees of
    freedom, and a chi-squared with two degrees of freedom has plenty of mass
    near zero, so the local estimate collapses to almost nothing several times
    in a three-hundred-frame run.  Every standard error computed from it
    collapses with it, the lower-confidence bound stops bounding anything, and
    the controller brakes at 4 m/s^2 for a lead holding a steady speed -- which
    is the original defect, reconstructed inside the fix for it.  A running
    estimate over the whole run has tens of degrees of freedom and does not do
    that.  Alpha of 0.1 settles in about thirty frames, 1.5 s."""

    @staticmethod
    def _ls_fit(
        samples: Sequence[Tuple[float, float]]
    ) -> Optional[Tuple[float, float, Optional[float]]]:
        """``(slope, Sxx, residual sigma)`` of ``y`` against ``t``.

        ``Sxx`` is the spread of the abscissae, so a caller can turn any noise
        estimate into a standard error with ``sigma / sqrt(Sxx)``.  The residual
        sigma is this fit's own opinion of the measurement noise and is None
        when there are too few degrees of freedom to have one.
        """
        n = len(samples)
        if n < 2:
            return None
        mt = sum(s[0] for s in samples) / n
        my = sum(s[1] for s in samples) / n
        sxx = sum((s[0] - mt) ** 2 for s in samples)
        if sxx <= 1e-12:
            return None
        slope = sum((s[0] - mt) * (s[1] - my) for s in samples) / sxx
        if n <= 2:
            return slope, sxx, None
        intercept = my - slope * mt
        resid = sum((s[1] - (intercept + slope * s[0])) ** 2 for s in samples)
        return slope, sxx, math.sqrt(max(0.0, resid / (n - 2)))

    def _note_noise(self, hist: Sequence[Tuple[float, float]]) -> None:
        """Update the running measurement-noise estimate from ``hist``.

        SECOND DIFFERENCES, not fit residuals.  The residuals of a straight line
        fitted to a range that is genuinely curving -- which is exactly what a
        braking lead produces -- are dominated by the curvature, not by the
        noise, so a noise estimate taken from them rises with the very signal it
        is supposed to help detect: fit the noise from those residuals and the
        controller concludes that a lead braking at 6 m/s^2 is a noisy lead
        holding station, refuses it the lead-deceleration credit, and drives
        into it.  The second difference ``r[i] - 2 r[i-1] + r[i-2]`` annihilates
        any straight line exactly and leaves a constant acceleration with only
        ``a dt^2`` -- 0.015 m at 6 m/s^2 on a 50 ms grid, four per cent of the
        noise this harness injects -- so it measures the sensor and not the
        manoeuvre.  Its variance is six times the measurement variance for
        independent samples, which is the ``/ 6``.

        The estimate is an EWMA over the whole run because the noise is a
        property of the sensor and one triple of samples is one degree of
        freedom.
        """
        if len(hist) < 3:
            return
        (t0, r0), (t1, r1), (t2, r2) = hist[-3], hist[-2], hist[-1]
        if abs((t2 - t1) - (t1 - t0)) > 1e-6:
            # Unequally spaced captures: the second difference is not a clean
            # noise probe across a frame overrun, so this triple is skipped.
            return
        var = (r2 - 2.0 * r1 + r0) ** 2 / 6.0
        if self._sigma_var is None:
            self._sigma_var = var
        else:
            self._sigma_var += self.SIGMA_EWMA_ALPHA * (var - self._sigma_var)
        self._sigma_range = math.sqrt(max(0.0, self._sigma_var))

    def _slope_se(self, sxx: float, sigma_local: Optional[float]) -> float:
        """Standard error of a slope fitted over samples with spread ``sxx``.

        ``sigma_local`` is this fit's own residual estimate and is used only
        when it is LARGER than the running one, so a window that has just seen
        something the running estimate has not is not ignored.
        """
        sigma = max(sigma_local or 0.0, self._sigma_range or 0.0)
        return sigma / math.sqrt(sxx) if sxx > 1e-12 else 0.0

    @classmethod
    def _ls_slope(cls, samples: Sequence[Tuple[float, float]]) -> Optional[float]:
        """Least-squares slope of ``y`` against ``t``, or None."""
        got = cls._ls_fit(samples)
        return None if got is None else got[0]

    def _update_range(
        self, tid: int, t_s: float, rng_m: float
    ) -> Optional[Tuple[float, float]]:
        """Add a range sample and return ``(closing estimate, confident closure)``.

        Two numbers because they answer two questions.  The ESTIMATE is the
        unbiased slope and is what the time-gap law needs, since that law is
        regulating a following speed and a biased estimate of the lead's speed
        makes it chase its own tail: subtract the uncertainty there and the ego
        keeps deciding it is 3 m/s too fast, however slowly it is going.  The
        CONFIDENT CLOSURE is the estimate less three standard errors and is what
        the braking law uses, because braking is irreversible and a closure that
        is indistinguishable from noise is not a reason to decelerate.

        Positive is closing.  A sample that jumps further than
        :data:`RANGE_JUMP_M` from where the current estimate predicted is a
        re-anchor, not motion: the history is dropped and the rate goes back to
        unknown until two fresh captures exist.  Differentiating such a jump is
        how a 10 m range correction becomes a 200 m/s closure and a phantom
        emergency.
        """
        hist = self._range_hist.setdefault(tid, [])
        if hist and t_s <= hist[-1][0] + 1e-12:
            # The same capture twice: a decision loop faster than the camera.
            if len(hist) < self.RATE_MIN_SAMPLES:
                return None
            fit = self._ls_fit(hist)
            if fit is None:
                return None
            se = self._slope_se(fit[1], None)
            return -fit[0], -fit[0] - self.CONFIDENCE_SIGMAS * se
        if hist:
            prev_t, prev_r = hist[-1]
            slope = self._ls_slope(hist)
            predicted = prev_r + (slope if slope is not None else 0.0) * (t_s - prev_t)
            if abs(rng_m - predicted) > self.RANGE_JUMP_M:
                hist = []
                self._range_hist[tid] = hist
        hist.append((t_s, rng_m))
        if len(hist) > self.RATE_WINDOW:
            del hist[0]
        if len(hist) < self.RATE_MIN_SAMPLES:
            return None
        fit = self._ls_fit(hist)
        if fit is None:
            return None
        slope, sxx, sigma_local = fit
        self._note_noise(hist)
        se = self._slope_se(sxx, None)
        return -slope, -slope - self.CONFIDENCE_SIGMAS * se

    def _lead_accel(self, tid: int, a_ego_mps2: float) -> float:
        """Confident lead acceleration, m/s^2, from the RAW range history.

        SPLIT HALVES, not a second differentiation of a smoothed series.  The
        obvious implementation -- differentiate the estimated lead speed, which
        is itself a differentiated range -- produces errors that are strongly
        CORRELATED between frames, because consecutive estimates share most of
        their samples.  A residual-based standard error then reports almost
        zero uncertainty for a trend that is entirely noise, and the controller
        confidently concludes that a lead holding a steady 20 m/s is braking at
        4 m/s^2.  That is not a tuning problem, it is the wrong estimator.

        Instead the window is split into two halves with NO SAMPLES IN COMMON, a
        slope is fitted to each, and the relative acceleration is the difference
        of the slopes over the gap between their centres.  The two slope errors
        are then independent, so ``sqrt(se_old^2 + se_new^2) / dt`` is an honest
        standard error, and the same lower-confidence-bound rule applies: credit
        only the deceleration that survives three of them.

        ``range'' = a_lead - a_ego``, so the ego's own acceleration -- known
        from the vehicle bus, not from the camera -- is added back.
        """
        hist = self._range_hist.get(tid) or []
        n = len(hist)
        half = self.RATE_MIN_SAMPLES
        if n < 2 * half:
            return 0.0
        old_s, new_s = hist[:half], hist[n - half :]
        f_old, f_new = self._ls_fit(old_s), self._ls_fit(new_s)
        if f_old is None or f_new is None:
            return 0.0
        t_old = sum(s[0] for s in old_s) / len(old_s)
        t_new = sum(s[0] for s in new_s) / len(new_s)
        span = t_new - t_old
        if span <= 1e-9:
            return 0.0
        a_rel = (f_new[0] - f_old[0]) / span
        se_old = self._slope_se(f_old[1], None)
        se_new = self._slope_se(f_new[1], None)
        se = math.sqrt(se_new * se_new + se_old * se_old) / span
        a_lead = a_rel + a_ego_mps2
        return max(-8.0, min(0.0, a_lead + self.ACCEL_CONFIDENCE_SIGMAS * se))

    def _update_ego_accel(self, t_s: float, v_ego: float) -> float:
        """The ego's own acceleration, differentiated from the vehicle bus.

        Not from the camera: the speed signal is the system's own
        proprioception and is orders of magnitude cleaner than a range.
        """
        hist = self._vego_hist
        if not hist or t_s > hist[-1][0] + 1e-12:
            hist.append((t_s, v_ego))
            if len(hist) > self.RATE_WINDOW:
                del hist[0]
        fit = self._ls_fit(hist)
        return 0.0 if fit is None else fit[0]

    # -------------------------------------------------------------- the laws

    def _required_decel(
        self, rng_m: float, closing_mps: float, v_ego: float, a_lead: float
    ) -> float:
        """Constant deceleration that keeps :data:`TARGET_CLEARANCE_M`, m/s^2.

        Textbook, from measured quantities only.  Two cases, because they are
        physically different problems: against a lead that is stopping the ego
        must come to rest inside the gap plus whatever the lead still travels;
        against one that is not, the ego only has to wash out the relative speed.
        """
        # The clearance target is capped at the room that still exists, exactly
        # as the oracle caps it.  Once the ego is already inside the standstill
        # clearance no deceleration can restore it, and demanding it anyway
        # turns the law into a step to full authority in the last metre of an
        # otherwise correct stop -- braking hard for a gap that is no longer
        # closing, which is disproportionate however it is dressed up.
        clearance = min(self.TARGET_CLEARANCE_M, max(0.0, rng_m - 0.3))
        room = max(0.05, rng_m - clearance)
        # First pass with no allowance, to find out how far the demand has to
        # travel; then charge half of that ramp's worth of closure.
        naive = (
            (max(0.0, closing_mps) ** 2) / (2.0 * room)
            if closing_mps > 0.05
            else 0.0
        )
        if a_lead < -0.5:
            naive = max(naive, (v_ego * v_ego) / (2.0 * room))
        t_ramp = min(
            self.RAMP_ALLOWANCE_S,
            max(0.0, naive - self._decel) / truth.EMERGENCY_JERK_MPS3,
        )
        usable = max(0.05, room - max(0.0, closing_mps) * 0.5 * t_ramp)
        if a_lead < -0.5:
            v_lead = max(0.0, v_ego - closing_mps)
            avail = max(0.05, usable + v_lead * v_lead / (2.0 * (-a_lead)))
            return min(
                DEFAULT_PLANT.max_brake_decel_mps2, (v_ego * v_ego) / (2.0 * avail)
            )
        if closing_mps <= 0.05:
            return 0.0
        return min(
            DEFAULT_PLANT.max_brake_decel_mps2,
            (closing_mps * closing_mps) / (2.0 * usable),
        )

    def _lateral(self, obs: Observation, dt_s: float) -> float:
        """Normalised steering.  Holds the last command while blind."""
        offset = obs.lateral_offset_m
        if offset is None:
            return self._steer
        if self._last_offset is not None and dt_s > 0.0:
            raw = (offset - self._last_offset) / dt_s
            self._offset_rate += 0.4 * (raw - self._offset_rate)
        self._last_offset = offset
        v = max(3.0, obs.ego.speed_mps)
        accel_cmd = -(
            2.0 * self.LAT_ZETA * self.LAT_WN * self._offset_rate
            + self.LAT_WN * self.LAT_WN * offset
        )
        delta = WHEELBASE_M * accel_cmd / (v * v)
        self._steer = max(-1.0, min(1.0, delta / MAX_ROAD_WHEEL_RAD))
        return self._steer

    # ------------------------------------------------------------------- API

    def step(
        self, obs: Observation, dt_s: float
    ) -> Tuple[ControlCommand, ControlCommand, ControlCommand, object]:
        """One frame.  Same signature as :meth:`StackUnderTest.step`."""
        self._last_speed_mps = obs.ego.speed_mps
        v_ego = obs.ego.speed_mps
        dt = dt_s if dt_s and dt_s > 0 else DEFAULT_PLANT.dt_s
        steering = self._lateral(obs, dt)
        a_ego = self._update_ego_accel(obs.t_s, v_ego)

        if not obs.perception.ok:
            self._blind += 1
            if self._blind > self.BLIND_HOLD_FRAMES:
                target = self.MRM_DECEL_MPS2
                state = SafetyState.MIN_RISK_MANEUVER
                reason = "minimum_risk_stop"
            else:
                target = self._decel
                state = SafetyState.LIMITED
                reason = "perception_dropout_hold"
            v_target = 0.0
        else:
            self._blind = 0
            lead = self._nearest_in_path(obs)
            if lead is None:
                self._miss_frames += 1
                if self._miss_frames <= self.MISS_HOLD_FRAMES:
                    a_avoid = self._held_avoid
                    reason = "detection_miss_hold" if self._held_avoid > 0.0 else "cruise"
                else:
                    a_avoid = 0.0
                    self._held_avoid = 0.0
                    reason = "cruise"
                v_target = self.cruise_speed_mps
                a_headway = 0.0
            else:
                self._miss_frames = 0
                rng = max(0.01, float(lead.distance_m))
                got = self._update_range(lead.track_id, obs.measurement_t_s, rng)
                if got is None:
                    a_avoid = 0.0
                    v_lead_est = v_ego
                    reason = "headway_no_rate"
                else:
                    rate, confident = got
                    v_lead_est = max(0.0, v_ego - rate)
                    a_lead = self._lead_accel(lead.track_id, a_ego)
                    # Gate on the confident bound, size on the estimate.
                    a_avoid = (
                        self._required_decel(rng, max(0.0, rate), v_ego, a_lead)
                        if confident > 0.05
                        else self._required_decel(rng, 0.0, v_ego, a_lead)
                    )
                    if a_avoid > 0.05:
                        a_avoid = min(
                            DEFAULT_PLANT.max_brake_decel_mps2,
                            a_avoid * (1.0 + self.DEMAND_MARGIN_FRAC)
                            + self.DEMAND_MARGIN_MPS2,
                        )
                    else:
                        a_avoid = 0.0
                    reason = "avoid" if a_avoid > 0.0 else "follow"
                self._held_avoid = a_avoid
                desired = self.STANDOFF_M + self.TIME_GAP_S * v_ego
                dv = max(0.0, min(self.HEADWAY_DV_MAX, self.HEADWAY_GAIN * (desired - rng)))
                v_target = max(0.0, min(self.cruise_speed_mps, v_lead_est - dv))
                a_headway = max(
                    0.0,
                    min(
                        truth.HEADWAY_DECEL_ALLOWANCE_MPS2,
                        self.SPEED_GAIN * (v_ego - v_target),
                    ),
                )
            target = max(a_avoid, a_headway)
            state = (
                SafetyState.LIMITED
                if target >= truth.COMFORT_DECEL_MPS2
                else SafetyState.NOMINAL
            )

        # Jerk limit.  The rising band comes from where the demand is HEADING:
        # a demand bound for emergency grade is a collision-avoidance action and
        # gets the emergency ceiling for the whole ramp, which is what the
        # judgement layer's own local-peak rule grants it.
        if target > self._decel:
            rate_limit = (
                truth.EMERGENCY_JERK_MPS3
                if target >= truth.EMERGENCY_DECEL_MPS2
                else truth.COMFORT_JERK_MPS3
            )
            self._decel = min(target, self._decel + rate_limit * dt)
        else:
            self._decel = max(target, self._decel - 12.0 * dt)

        brake = max(0.0, min(1.0, self._decel / DEFAULT_PLANT.max_brake_decel_mps2))
        throttle = 0.0
        if brake <= 1e-6 and obs.perception.ok and v_target > v_ego + 0.05:
            throttle = max(0.0, min(1.0, self.THROTTLE_GAIN * (v_target - v_ego)))
            brake = 0.0
        self._last_brake = brake

        command = ControlCommand(throttle, brake, steering)
        result = ArbitrationResult(
            command=command, state=state, violations=[], reason=reason
        )
        result.plan_reason = reason
        result.plan_target = v_target
        return command, command, command, result

    def _nearest_in_path(self, obs: Observation):
        """The closest reported track whose footprint overlaps the ego's own.

        It does NOT read ``TrackedObject.in_ego_lane``.  That flag is the lane
        estimator's opinion, and a lane fit that has slipped a metre towards the
        next lane reports a car that is nowhere near the ego as being in front
        of it -- which is a lane error CREATING a hazard, the mirror image of a
        lane error hiding one.  The geometry that decides whether a collision is
        possible is the object's lateral offset from the EGO, which the detector
        measures from the box's position in the image and which no lane model
        can move.  Half the ego plus half the object is 1.8 m; a car one lane
        over is at 3.5 m and is not in the way however confident the lane fit is.
        """
        best = None
        for t in obs.tracks:
            if t.distance_m is None or t.distance_m <= 0.0:
                continue
            lateral = getattr(t, "lateral_offset_m", None)
            if lateral is not None:
                half_w = 0.5 * CAR_WIDTH_M
                box = getattr(t, "box", None)
                if box is not None and t.distance_m > 0.0:
                    half_w = max(
                        0.3,
                        0.5 * (box.x2 - box.x1) * float(t.distance_m) / CAMERA_FOCAL_PX,
                    )
                if abs(float(lateral)) > EGO_HALF_WIDTH_M + half_w:
                    continue
            if best is None or t.distance_m < best.distance_m:
                best = t
        return best

    def fail_safe(self, t_s: float, ego_speed_mps: float, dt_s: float) -> ControlCommand:
        """The command written when the loop ends: throttle off, brake held.

        ADAS-DEC-21.  Leaving the loop is not a command, so a defined one is
        emitted: never any throttle, and never less brake than the loop's last
        frame was applying, with a floor at the comfort deceleration whenever the
        vehicle is still moving.
        """
        floor = (
            truth.COMFORT_DECEL_MPS2 / DEFAULT_PLANT.max_brake_decel_mps2
            if ego_speed_mps > STANDSTILL_MPS
            else 0.0
        )
        return ControlCommand(0.0, max(self._last_brake, floor), 0.0)

    def shutdown(self, t_s: float, dt_s: float, hold_steps: int = 4) -> List[ControlCommand]:
        """The commands written after the frame loop ends."""
        return [
            self.fail_safe(t_s + (i + 1) * dt_s, self._last_speed_mps, dt_s)
            for i in range(hold_steps)
        ]


def reference_stack(scenario: "Scenario") -> ReferenceController:
    """A :class:`ReferenceController` configured for ``scenario``."""
    return ReferenceController(
        cruise_speed_mps=(
            scenario.cruise_speed_mps
            if scenario.cruise_speed_mps is not None
            else scenario.ego_speed_mps
        ),
        spec=scenario.stack,
    )


# --------------------------------------------------------------------------- #
# Expectations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Expectation:
    """What the system is required to do.  Every field is optional except the
    two that are always on: no collision, and no disengagement.

    Attributes:
        no_collision: Contact is a failure whenever the oracle says it was
            avoidable.  An unavoidable collision is not held against the system.
        min_clearance_m: A floor on the true gap.  Use it where stopping short
            is the whole point of the scenario.
        must_intervene: The scenario contains a genuine emergency, so the system
            must command at least :data:`oracle.EMERGENCY_DECEL_MPS2` no later
            than the oracle's last avoidance frame.
        forbid_emergency_intervention: The scenario contains no emergency, so
            the system must never command emergency-grade deceleration and must
            never enter MIN_RISK_MANEUVER while perception is healthy.
        max_commanded_decel_mps2: Hard ceiling on the commanded deceleration
            across the whole run.
        max_safety_state: The worst safety state the run may reach.
        recover_within_frames: After the oracle's ``hazard_clear_frame`` the
            system must be back in NOMINAL within this many frames, and stay
            there.
        min_speed_floor_mps: The ego's true speed must never fall below this.
            This is what makes "it braked for nothing" measurable in metres per
            second rather than in state labels.
        min_final_speed_mps: Lower bound on the ego's speed at the last frame.
        max_final_speed_mps: Upper bound on the ego's speed at the last frame.
            Use it where coming to a stop IS the required behaviour, such as a
            permanent sensor loss.
        max_abs_lateral_offset_m: The ego must stay within this distance of the
            lane centre.
        require_fail_safe_exit: After the loop ends the emitted commands must
            release the throttle and must not release the brake.
        unjustified_brake_frames_allowed: Number of frames on which the
            commanded deceleration may exceed the oracle's justified ceiling
            WHILE an intervention is warranted (the ``disproportionate_brake``
            finding).  Non-zero only where a transient is genuinely defensible,
            and the reason must be in the scenario's ``physics`` string.  It
            never excuses a ``phantom_intervention``: braking with no warrant at
            all is not a transient.
        unwarranted_brake_frames_allowed: Number of frames on which the
            commanded deceleration may exceed
            :data:`oracle.HEADWAY_DECEL_ALLOWANCE_MPS2` while NOTHING was
            required (the ``unwarranted_brake`` finding -- the sub-emergency
            band).  Default zero.  Raise it only where the scenario itself
            deliberately lies to the system about the range, and say so in
            ``physics``.
        allowed_self_reports: Prefixes of ``ArbitrationResult.violations`` this
            scenario tolerates even though :data:`ASSERTABLE_SELF_REPORTS`
            matches them.  Every entry needs an argument in ``physics``: the
            system said something was wrong with its own output, and the
            scenario is claiming it was right to and that it does not matter.
        primary_must_stop: The PRIMARY path (planner + controller) must reach at
            least :data:`oracle.EMERGENCY_DECEL_MPS2` of its own accord, before
            the arbiter is consulted.  Set on scenarios that exist to show the
            primary path is not relying on the backstop.
        arbiter_must_veto_brake_below_mps2: The arbiter's own command must never
            exceed this deceleration.  Set on scenarios where the primary path
            is jamming the brake on and the arbiter is required to VETO it
            rather than pass it through, which is the difference between an
            authority and a wire.
        max_jerk_mps3: Override on the demanded-jerk ceiling.  Leave None to use
            :func:`oracle.jerk_limit_mps3`, which picks the comfort or the
            emergency band per frame from the true kinematics.  An override is
            an admission that the scenario is special; justify it in ``physics``.
    """

    no_collision: bool = True
    min_clearance_m: Optional[float] = None
    must_intervene: bool = False
    forbid_emergency_intervention: bool = False
    max_commanded_decel_mps2: Optional[float] = None
    max_safety_state: Optional[SafetyState] = None
    recover_within_frames: Optional[int] = None
    min_speed_floor_mps: Optional[float] = None
    min_final_speed_mps: Optional[float] = None
    max_final_speed_mps: Optional[float] = None
    max_abs_lateral_offset_m: Optional[float] = None
    require_fail_safe_exit: bool = False
    unjustified_brake_frames_allowed: int = 0
    unwarranted_brake_frames_allowed: int = 0
    allowed_self_reports: Tuple[str, ...] = ()
    primary_must_stop: bool = False
    arbiter_must_veto_brake_below_mps2: Optional[float] = None
    max_jerk_mps3: Optional[float] = None


_STATE_ORDER = {
    SafetyState.NOMINAL: 0,
    SafetyState.LIMITED: 1,
    SafetyState.MIN_RISK_MANEUVER: 2,
    SafetyState.DISENGAGE: 3,
}


# --------------------------------------------------------------------------- #
# Scenario
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scenario:
    """One acceptance case.

    Attributes:
        name: Stable identifier; used as the pytest id and the JSON key.
        summary: One line describing the world.
        physics: The argument for the expected outcome.  Numbers, not adjectives.
            The report prints this next to the verdict so that a failure can be
            read without opening the source.
        guards: Which historical failure this scenario is the regression test
            for, or "" for an ordinary case.
        frames: Number of simulated frames (at :data:`plant.DT_S`).
        ego_speed_mps: Initial ego speed.
        cruise_speed_mps: The planner's cruise target.  Defaults to the initial
            ego speed, which is the ordinary "set cruise and drive" case.
        lead: The lead vehicle script, or None for an empty road.  The oracle
            judges avoidability against this object.
        others: Any number of ADDITIONAL objects, simulated and reported exactly
            like the lead.  The plant has accepted them and published
            ``WorldState.objects``, ``in_path_objects`` and ``min_in_path_gap_m``
            for several revisions and nothing could reach them, because this
            class had no field for them: a capability the specification cannot
            express is not coverage.  Use them for a next-lane vehicle beside a
            real lead, a queue, or an object the ego must NOT confuse with the
            one that matters.
        perception: Sensor characteristics.
        road: Road geometry.
        expect: The requirements.
        stack: Which layers of the longitudinal path are real.  The default runs
            all three; see :class:`StackSpec`.
        terminate_during_run: Run the shutdown path after the loop.
    """

    name: str
    summary: str
    physics: str
    frames: int
    ego_speed_mps: float
    expect: Expectation
    guards: str = ""
    cruise_speed_mps: Optional[float] = None
    lead: Optional[LeadSpec] = None
    others: Tuple[LeadSpec, ...] = ()
    perception: PerceptionSpec = PERFECT_PERCEPTION
    road: RoadSpec = field(default_factory=straight_road)
    initial_lateral_offset_m: float = 0.0
    terminate_during_run: bool = False
    config: PlantConfig = DEFAULT_PLANT
    stack: StackSpec = DEFAULT_STACK


@dataclass
class Finding:
    """One reason a scenario failed."""

    code: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return "%s: %s" % (self.code, self.detail)


@dataclass
class ScenarioResult:
    """Everything the report and the tests need about one run."""

    scenario: Scenario
    history: List[WorldState]
    records: List[FrameRecord]
    verdict: truth.OracleVerdict
    exit_commands: List[ControlCommand]
    findings: List[Finding]

    @property
    def passed(self) -> bool:
        return not self.findings

    @property
    def metrics(self) -> Dict[str, object]:
        """Compact numeric summary, safe to serialise."""
        decels = [r.commanded_decel_mps2 for r in self.records]
        speeds = [r.true.ego_v_mps for r in self.records]
        states = [r.safety_state.value for r in self.records]
        cmd_jerk = [j for j in commanded_jerk_series(self.records, self.scenario.config)
                    if j is not None]
        got_jerk = [j for j in achieved_jerk_series(self.records, self.scenario.config)
                    if j is not None]
        self_reports = collect_assertable_self_reports(
            self.records, self.scenario.expect.allowed_self_reports
        )
        return {
            "frames": len(self.records),
            "stack": self.scenario.stack.label,
            "min_gap_m": _round(self.verdict.min_gap_m),
            "collided": self.verdict.collided,
            "oracle_emergency": self.verdict.emergency,
            "oracle_first_emergency_frame": self.verdict.first_emergency_frame,
            "oracle_last_avoidance_frame": self.verdict.last_avoidance_frame,
            "oracle_hazard_clear_frame": self.verdict.hazard_clear_frame,
            "oracle_avoidable": self.verdict.avoidable,
            "oracle_earliest_actionable_frame": self.verdict.earliest_actionable_frame,
            "contact": (
                {
                    "frame": self.verdict.contact.frame,
                    "t_s": _round(self.verdict.contact.t_s),
                    "gap_m": _round(self.verdict.contact.gap_m),
                    "closing_mps": _round(self.verdict.contact.closing_mps),
                    "ego_v_mps": _round(self.verdict.contact.ego_v_mps),
                    "object_v_mps": _round(self.verdict.contact.object_v_mps),
                    "label": self.verdict.contact.label,
                }
                if self.verdict.contact is not None
                else None
            ),
            "affordable_decision_latency_s": _round(
                feasibility(self.scenario).affordable_decision_latency_s
            ),
            "scenario_feasible": feasibility(self.scenario).feasible,
            "max_commanded_decel_mps2": _round(max(decels) if decels else 0.0),
            "max_primary_decel_mps2": _round(
                max((r.raw_decel_mps2 for r in self.records), default=0.0)
            ),
            "max_arbiter_decel_mps2": _round(
                max((r.arbiter_decel_mps2 for r in self.records), default=0.0)
            ),
            "max_commanded_jerk_mps3": _round(max(cmd_jerk) if cmd_jerk else 0.0),
            "max_achieved_jerk_mps3": _round(max(got_jerk) if got_jerk else 0.0),
            "pedal_conflict_frames": sum(1 for r in self.records if r.pedal_conflict),
            "self_reported_violations": {
                k: len(v) for k, v in sorted(self_reports.items())
            },
            "max_required_decel_mps2": _round(
                max((d for d in self.verdict.required_decel if math.isfinite(d)), default=0.0)
            ),
            "min_ego_speed_mps": _round(min(speeds) if speeds else 0.0),
            "final_ego_speed_mps": _round(speeds[-1] if speeds else 0.0),
            "max_abs_lateral_offset_m": _round(
                max((abs(r.true.lateral_offset_m) for r in self.records), default=0.0)
            ),
            "worst_safety_state": max(
                (r.safety_state for r in self.records),
                key=lambda s: _STATE_ORDER[s],
                default=SafetyState.NOMINAL,
            ).value,
            "state_histogram": {s: states.count(s) for s in sorted(set(states))},
            "exit_commands": [
                {"throttle": _round(c.throttle), "brake": _round(c.brake)}
                for c in self.exit_commands
            ],
        }


def _round(x: float, places: int = 3) -> float:
    if x is None or not math.isfinite(x):
        return float(x) if x is not None else 0.0
    return round(float(x), places)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def run(scenario: Scenario, sut: Optional[StackUnderTest] = None) -> ScenarioResult:
    """Run one scenario closed loop and judge it.

    The loop is: true state -> sensor -> system -> actuated command -> plant.
    Nothing consults a clock, and the only randomness is the sensor's seeded
    generator, so two runs of the same scenario are identical.
    """
    cfg = scenario.config
    dt = cfg.dt_s
    plant = Plant(
        ego_speed_mps=scenario.ego_speed_mps,
        lead=scenario.lead,
        road=scenario.road,
        config=cfg,
        initial_lateral_offset_m=scenario.initial_lateral_offset_m,
        others=tuple(scenario.others),
    )
    sensor = Sensor(scenario.perception, cfg)
    system = sut or StackUnderTest(
        cruise_speed_mps=(
            scenario.cruise_speed_mps
            if scenario.cruise_speed_mps is not None
            else scenario.ego_speed_mps
        ),
        spec=scenario.stack,
    )

    history: List[WorldState] = []
    records: List[FrameRecord] = []
    state = plant.state
    for _ in range(scenario.frames):
        history.append(state)
        obs = sensor.observe(state)
        command, raw, arbiter_cmd, result = system.step(obs, dt)
        records.append(
            FrameRecord(
                frame=state.frame,
                t_s=state.t_s,
                true=state,
                perception_ok=obs.perception.ok,
                detected=bool(obs.tracks),
                plan_reason=getattr(result, "plan_reason", "") or "",
                plan_target_mps=float(getattr(result, "plan_target", 0.0) or 0.0),
                raw_command=raw,
                command=command,
                safety_state=getattr(result, "state", SafetyState.MIN_RISK_MANEUVER),
                violations=list(getattr(result, "violations", []) or []),
                arbiter_command=arbiter_cmd,
            )
        )
        state = plant.step(command.throttle, command.brake, command.steering)
        if plant.contacted:
            # THE RUN IS OVER.  The contact frame is kept in the history so the
            # oracle can read the impact speed off it, and nothing after it is
            # simulated: two bodies that have collided are not still driving,
            # and every finding derived from the frames that follow is
            # arithmetic about a world that does not exist.  Before this,
            # lead_brakes_6mps2_ego20_at_20m hit the lead at frame 71, ran on to
            # frame 300, and the report printed a "minimum true gap" of
            # -38.07 m -- the distance by which the ego had driven THROUGH it.
            history.append(state)
            break

    exit_commands: List[ControlCommand] = []
    if scenario.terminate_during_run:
        exit_commands = system.shutdown(state.t_s, dt)

    verdict = truth.judge(history, scenario.lead, scenario.road, cfg, scenario.perception)
    findings = evaluate(scenario, records, verdict, exit_commands)
    return ScenarioResult(
        scenario=scenario,
        history=history,
        records=records,
        verdict=verdict,
        exit_commands=exit_commands,
        findings=findings,
    )


# --------------------------------------------------------------------------- #
# Satisfiability: is this scenario passable by ANY correct system?
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Feasibility:
    """Whether a scenario's requirement can be met, and by how much time.

    Computed WITHOUT running the system under test: a neutral ego holds its
    cruise, the sensor model says when the hazard could first be acted on, and
    the plant says how long full authority can be deferred and still hold the
    clearance the scenario demands.  Nothing here depends on what the arbiter
    does, which is what makes it usable as an acceptance criterion for the
    harness itself.

    Attributes:
        name: The scenario's name.
        required_clearance_m: The clearance the expectation demands (contact,
            when it only forbids collision).
        ideal_clearance_m: Best clearance a ZERO-latency system could hold.
            The arithmetic bound; no system can beat it.
        best_clearance_m: Best clearance a system that pays the unavoidable
            latency can hold.  This is the number that matters.
        earliest_actionable_frame: First frame on which the measurements could
            support the decision; see
            :func:`oracle.earliest_actionable_frame`.
        minimum_latency_s: That frame in seconds -- the reaction time no
            correct system can get below.
        total_budget_s: Longest decision latency, measured from frame 0, that
            still holds ``required_clearance_m``.
        affordable_decision_latency_s: ``total_budget_s - minimum_latency_s``.
            THE number: how much time the system has to think, on top of the
            time the pipeline has already spent.  Negative means the scenario
            demands a reaction before the information exists, and the only way
            to pass it is to brake on a prior -- which is the phantom braking
            this harness punishes elsewhere.  A scenario like that is
            mis-specified and must be moved or deleted.
        feasible: ``affordable_decision_latency_s >= 0``.
    """

    name: str
    required_clearance_m: float
    ideal_clearance_m: float
    best_clearance_m: float
    earliest_actionable_frame: Optional[int]
    minimum_latency_s: float
    total_budget_s: float
    affordable_decision_latency_s: float
    feasible: bool

    @property
    def summary(self) -> str:
        """One line for the report and for a finding's detail text."""
        if self.earliest_actionable_frame is None:
            return "no hazard: nothing to be in time for"
        return (
            "requires %.2f m; a zero-latency system holds %.2f m and a real one "
            "%.2f m. The hazard is first actionable at frame %d (%.3f s: %.3f s "
            "of sense latency and rate observability), full authority may be "
            "deferred until %.3f s, so the affordable decision latency is "
            "%+.3f s"
            % (
                self.required_clearance_m,
                self.ideal_clearance_m,
                self.best_clearance_m,
                self.earliest_actionable_frame,
                self.minimum_latency_s,
                self.minimum_latency_s,
                self.total_budget_s,
                self.affordable_decision_latency_s,
            )
        )


_FEASIBILITY_CACHE: Dict[Tuple[object, ...], Feasibility] = {}


def _neutral_history(scenario: "Scenario") -> List[WorldState]:
    """The scenario with the ego holding its cruise and commanding nothing.

    The reference trajectory against which "when could this first be known?" is
    asked, so that the answer is a property of the SCENARIO and not of whatever
    the system under test happened to do.
    """
    plant = Plant(
        ego_speed_mps=scenario.ego_speed_mps,
        lead=scenario.lead,
        road=scenario.road,
        config=scenario.config,
        initial_lateral_offset_m=scenario.initial_lateral_offset_m,
        others=tuple(scenario.others),
    )
    out = [plant.state]
    for _ in range(scenario.frames - 1):
        out.append(plant.step(0.0, 0.0, 0.0))
        if plant.contacted:
            break
    return out


def feasibility(scenario: "Scenario") -> Feasibility:
    """Whether ``scenario``'s clearance requirement is physically satisfiable.

    Cached per scenario, because the bisection runs about forty full-authority
    simulations and the answer cannot change between calls.
    """
    exp = scenario.expect
    target = (
        float(exp.min_clearance_m)
        if exp.min_clearance_m is not None
        else (truth.CONTACT_GAP_M + 1e-6 if exp.no_collision else float("-inf"))
    )
    key = (scenario.name, scenario.frames, scenario.ego_speed_mps, target)
    got = _FEASIBILITY_CACHE.get(key)
    if got is not None:
        return got

    cfg = scenario.config
    if scenario.lead is None or not math.isfinite(target):
        out = Feasibility(
            name=scenario.name,
            required_clearance_m=target,
            ideal_clearance_m=float("inf"),
            best_clearance_m=float("inf"),
            earliest_actionable_frame=None,
            minimum_latency_s=0.0,
            total_budget_s=float("inf"),
            affordable_decision_latency_s=float("inf"),
            feasible=True,
        )
        _FEASIBILITY_CACHE[key] = out
        return out

    history = _neutral_history(scenario)
    by_frame = {s.frame: s for s in history}
    actionable = truth.earliest_actionable_frame(history, scenario.perception, cfg)
    hazard = next((s for s in history if s.lead_present), None)
    start = by_frame.get(actionable) if actionable is not None else None
    if hazard is None or start is None:
        out = Feasibility(
            name=scenario.name,
            required_clearance_m=target,
            ideal_clearance_m=float("inf"),
            best_clearance_m=float("inf"),
            earliest_actionable_frame=actionable,
            minimum_latency_s=0.0,
            total_budget_s=float("inf"),
            affordable_decision_latency_s=float("inf"),
            feasible=True,
        )
        _FEASIBILITY_CACHE[key] = out
        return out

    # The clock starts when the hazard EXISTS and the decision may be taken from
    # the first frame it can be MEASURED.  Both are properties of the scenario,
    # not of any system, because the neutral ego holds its cruise throughout.
    min_latency = max(0.0, (start.frame - hazard.frame) * cfg.dt_s)
    others = tuple(scenario.others)
    ideal = truth.full_braking_min_gap(
        hazard, scenario.lead, scenario.road, cfg, others=others
    )
    best = truth.full_braking_min_gap(
        start, scenario.lead, scenario.road, cfg, others=others
    )
    afford = truth.braking_budget_s(
        start, scenario.lead, scenario.road, cfg, clearance_m=target, others=others
    )
    budget = afford + min_latency if math.isfinite(afford) else afford
    out = Feasibility(
        name=scenario.name,
        required_clearance_m=target,
        ideal_clearance_m=ideal,
        best_clearance_m=best,
        earliest_actionable_frame=actionable,
        minimum_latency_s=min_latency,
        total_budget_s=budget,
        affordable_decision_latency_s=afford,
        feasible=afford >= -1e-9,
    )
    _FEASIBILITY_CACHE[key] = out
    return out


SCENARIO_DEFECT_CODES = frozenset(
    {
        "no_frames",
        "collided_unavoidable",
        "infeasible_clearance",
        "unactionable_scenario",
    }
)
"""Findings that indict the SCENARIO rather than the system under test.

A harness that reports "you failed to stop" for a stop that no vehicle could
make is manufacturing bug reports out of its own arithmetic, and it is worse
than useless: the only way to make such a scenario green is the misbehaviour
this specification punishes everywhere else -- braking on a prior instead of a
measurement.  These four codes say so out loud.  They still FAIL the scenario,
because a mis-specified acceptance case is a defect that has to be fixed; they
just name the right culprit.
"""


# --------------------------------------------------------------------------- #
# Judgement
# --------------------------------------------------------------------------- #


def evaluate(
    scenario: Scenario,
    records: Sequence[FrameRecord],
    verdict: truth.OracleVerdict,
    exit_commands: Sequence[ControlCommand],
) -> List[Finding]:
    """Compare what happened against what the oracle and the expectation demand.

    Returns an empty list when the scenario passed.  Each finding names the
    diagnosis and the frame, so a regression report is actionable without
    re-running anything.
    """
    exp = scenario.expect
    findings: List[Finding] = []
    if not records:
        return [Finding("no_frames", "the scenario produced no frames")]

    # ---------------------------------------------------------- collision ---
    ev = verdict.contact
    impact = (
        "impact at %.2f m/s (ego %.2f m/s into object %d, %r, at %.2f m/s)"
        % (ev.closing_mps, ev.ego_v_mps, ev.object_index, ev.label, ev.object_v_mps)
        if ev is not None
        else "impact speed not recorded"
    )
    if exp.no_collision and verdict.collided:
        feas = feasibility(scenario)
        if verdict.avoidable is False:
            findings.append(
                Finding(
                    "collided_unavoidable",
                    "contact at frame %d; %s. The oracle says it was ALREADY UNAVOIDABLE on "
                    "the first frame a real system could have acted (frame %s), so this "
                    "scenario is mis-specified: %s. Passing it would require braking before "
                    "the measurement exists, which is the phantom this harness punishes "
                    "elsewhere -- move the case out or delete it."
                    % (
                        verdict.collision_frame,
                        impact,
                        verdict.earliest_actionable_frame,
                        feas.summary,
                    ),
                )
            )
        else:
            findings.append(
                Finding(
                    "collided",
                    "contact at frame %d (t=%.2f s); %s. Full-authority braking was still "
                    "sufficient up to frame %s, and a correct system had %+.3f s of decision "
                    "latency to spare (%s)."
                    % (
                        verdict.collision_frame,
                        records[min(verdict.collision_frame, len(records) - 1)].t_s,
                        impact,
                        verdict.last_avoidance_frame,
                        feas.affordable_decision_latency_s,
                        feas.summary,
                    ),
                )
            )

    if exp.min_clearance_m is not None:
        feas = feasibility(scenario)
        # Two independent questions, and the old code could only ask one.  "Is
        # this requirement reachable at all?" indicts the SCENARIO; "did the
        # system get as close to the limit as a correct one would?" indicts the
        # SYSTEM.  When a requirement is impossible the second question is still
        # worth asking, but it has to be asked against the reachable clearance
        # rather than against the impossible one, or every run of a
        # mis-specified case reports a system failure it did not commit.
        # In the infeasible branch the comparison is against a THEORETICAL
        # optimum -- a bang-bang brake committed on the first actionable frame
        # -- which no closed-loop law can match exactly.  One frame of travel is
        # conceded, because a shortfall smaller than the harness's own time
        # quantisation cannot be attributed to the system.
        reachable = (
            exp.min_clearance_m
            if feas.feasible
            else feas.best_clearance_m - scenario.ego_speed_mps * scenario.config.dt_s
        )
        if not feas.feasible:
            findings.append(
                Finding(
                    "infeasible_clearance",
                    "this scenario demands %.2f m of clearance that NO SYSTEM CAN HOLD: %s. "
                    "The requirement is inside the reaction time of the pipeline the system "
                    "under test is given, so the only way to satisfy it is to command full "
                    "authority before the closing rate has been measured -- exactly the "
                    "unwarranted braking this specification treats as a failure everywhere "
                    "else. The run itself reached %.2f m. This is a defect in the SCENARIO: "
                    "move the obstacle out to at least the range at which %.2f m is "
                    "reachable, or lower the requirement to the %.2f m that is."
                    % (
                        exp.min_clearance_m,
                        feas.summary,
                        verdict.min_gap_m,
                        exp.min_clearance_m,
                        max(0.0, feas.best_clearance_m),
                    ),
                )
            )
        if verdict.min_gap_m < reachable - 1e-6:
            findings.append(
                Finding(
                    "clearance",
                    "minimum true gap %.2f m, against the %.2f m %s. %s."
                    % (
                        verdict.min_gap_m,
                        reachable,
                        "this scenario requires"
                        if feas.feasible
                        else "a correct system would still have held here (the %.2f m the "
                        "scenario asks for is not reachable by anything)"
                        % exp.min_clearance_m,
                        feas.summary,
                    ),
                )
            )

    # ------------------------------------------------------- intervention ---
    first_emergency_cmd = next(
        (r.frame for r in records if r.commanded_decel_mps2 >= truth.EMERGENCY_DECEL_MPS2),
        None,
    )
    if exp.must_intervene:
        peak_cmd = max(r.commanded_decel_mps2 for r in records)
        if not verdict.emergency:
            # The system slowed early enough that the true requirement never
            # reached emergency grade.  That is the DESIRED outcome, not a
            # failure -- all that is required is that it responded at all.
            if peak_cmd < truth.NEGLIGIBLE_DECEL_MPS2:
                findings.append(
                    Finding(
                            "no_response",
                        "a lead was present and closing but the system never commanded even "
                        "%.1f m/s^2 (peak %.2f m/s^2). The primary path's last plan asked for "
                        "%.2f m/s (%s)."
                        % (
                            truth.NEGLIGIBLE_DECEL_MPS2,
                            peak_cmd,
                            records[-1].plan_target_mps,
                            records[-1].plan_reason or "no reason given",
                        ),
                    )
                )
        elif first_emergency_cmd is None:
            # The requirement can be +inf on every frame -- an obstacle that was
            # never avoidable at all -- so a plain max() over the finite entries
            # has no elements to take.  Report the infinity instead of crashing:
            # "the physics demanded more than the vehicle has and the system
            # asked for none of it" is the single most damning verdict this
            # harness can return, and it must survive being printed.
            finite = [d for d in verdict.required_decel if math.isfinite(d)]
            need = max(finite) if finite else float("inf")
            findings.append(
                Finding(
                    "missed_intervention",
                    "the oracle required %s from frame %s and the system never commanded "
                    "%.1f m/s^2 (peak %.2f m/s^2)"
                    % (
                        ("more than the vehicle's %.1f m/s^2 authority"
                         % scenario.config.max_brake_decel_mps2)
                        if math.isinf(need)
                        else "%.2f m/s^2" % need,
                        verdict.first_emergency_frame,
                        truth.EMERGENCY_DECEL_MPS2,
                        max(r.commanded_decel_mps2 for r in records),
                    )
                    + ". At the first frame the oracle required it, the plan was targeting "
                    "%.2f m/s (%s)."
                    % (
                        records[
                            min(verdict.first_emergency_frame or 0, len(records) - 1)
                        ].plan_target_mps,
                        records[
                            min(verdict.first_emergency_frame or 0, len(records) - 1)
                        ].plan_reason
                        or "no reason given",
                    ),
                )
            )
        elif verdict.last_avoidance_frame is not None:
            # A command cannot precede the measurement that provokes it.  The
            # deadline a system can be held to is the LATER of "the last frame
            # from which braking still works" and "the first frame on which the
            # hazard could be acted on"; holding it to the earlier of the two is
            # asking it to be clairvoyant, and it is how a scenario placed 1 m
            # inside the zero-latency boundary produced a lateness report
            # against a system that reacted on the very first frame it could.
            actionable = verdict.earliest_actionable_frame
            deadline = verdict.last_avoidance_frame
            if actionable is not None and actionable > deadline:
                findings.append(
                    Finding(
                        "unactionable_scenario",
                        "full-authority braking stops working after frame %d, but the hazard "
                        "cannot be acted on before frame %d: there is no frame on which any "
                        "system could have intervened in time. %s. This is a defect in the "
                        "SCENARIO, not in the system -- the case has to move to a range where "
                        "the deadline is later than the first actionable frame, or be deleted."
                        % (
                            deadline,
                            actionable,
                            feasibility(scenario).summary,
                        ),
                    )
                )
            elif first_emergency_cmd > deadline:
                findings.append(
                    Finding(
                        "late_intervention",
                        "first emergency-grade command at frame %d, %d frames (%.2f s) after "
                        "the last frame from which full braking still avoided contact (frame "
                        "%d). The hazard was actionable from frame %s, so %s frame(s) of "
                        "that delay was the system's own."
                        % (
                            first_emergency_cmd,
                            first_emergency_cmd - deadline,
                            (first_emergency_cmd - deadline) * scenario.config.dt_s,
                            deadline,
                            actionable,
                            (first_emergency_cmd - actionable)
                            if actionable is not None
                            else "all",
                        ),
                    )
                )

    # ------------------------------------------------------------ phantom ---
    # A vehicle already at a standstill holding its brake is not intervening, and
    # neither is one finishing a stop that WAS warranted when it began.  Both are
    # exempt: without the first, "stopped safely behind a parked car and stayed
    # stopped" scores as continuous unjustified braking; without the second, every
    # successful intervention is scored as a phantom on the way out of it, because
    # a brake that works makes its own justification disappear.
    exempt, blind_manoeuvre = _manoeuvre_exemptions(records, verdict)
    phantoms: List[FrameRecord] = []
    unwarranted: List[FrameRecord] = []
    unwarranted_states: List[FrameRecord] = []
    excessive: List[FrameRecord] = []
    for i, r in enumerate(records):
        if r.true.ego_v_mps <= STANDSTILL_MPS or not r.perception_ok:
            continue
        if blind_manoeuvre[i]:
            # A stop the system started because it had gone blind, or the tail
            # of one still being released now that it can see again.  The oracle
            # has no requirement to measure it against; the recovery and final
            # speed assertions are what police it.
            continue
        quiet = verdict.is_quiet(r.frame) and not exempt[i]
        # A STATE LABEL WITH NO ACTUATION IS NOT AN INTERVENTION.  These two used
        # to be one test -- "commanded_decel >= 3.5 OR state is MRM" -- which
        # diagnosed a reference stack with a phantom brake for entering a state
        # while commanding nothing.  They are different defects with different
        # consequences: one decelerates the vehicle and can cause a rear-end
        # collision behind, the other mis-classifies the world and will hand the
        # vehicle back or brake on the next frame.  Both are findings; they are
        # not the same finding.
        emergency_brake = r.commanded_decel_mps2 >= truth.EMERGENCY_DECEL_MPS2
        emergency_state = r.safety_state is SafetyState.MIN_RISK_MANEUVER
        if quiet:
            if emergency_state and not emergency_brake:
                unwarranted_states.append(r)
            if emergency_brake:
                phantoms.append(r)
            elif r.commanded_decel_mps2 > truth.HEADWAY_DECEL_ALLOWANCE_MPS2 + 1e-6:
                # The SUB-EMERGENCY band.  Nothing in the true world required any
                # braking, and the system is braking harder than a headway law
                # ever needs to -- but below AEB grade, so the phantom test above
                # cannot see it.  This is the band a "fix" for phantom braking
                # retreats into: the MRM stops, the 8 m/s^2 stops, and the
                # vehicle is still dragged down the motorway for no reason.
                unwarranted.append(r)
        elif r.commanded_decel_mps2 > verdict.justified_decel_mps2(r.frame) + 1e-6:
            excessive.append(r)

    if phantoms:
        first = phantoms[0]
        peak = max(phantoms, key=lambda r: r.commanded_decel_mps2)
        findings.append(
            Finding(
                "phantom_intervention",
                "EMERGENCY-GRADE BRAKING ACTUATED on %d frame(s) with no hazard; first at "
                "frame %d (state=%s, brake=%.2f = %.1f m/s^2, true gap %.1f m, true closing "
                "%+.2f m/s, true requirement %.2f m/s^2), peak %.2f m/s^2 at frame %d. This "
                "is the actuated half of the phantom: the vehicle really did decelerate, and "
                "a follower keeping a 2 s gap and taking 1 s to react can absorb 5 m/s^2 and "
                "cannot absorb 8, so it does not avoid a collision, it manufactures one "
                "behind."
                % (
                    len(phantoms),
                    first.frame,
                    first.safety_state.value,
                    first.command.brake,
                    first.commanded_decel_mps2,
                    first.true.gap_m,
                    first.true.closing_mps,
                    verdict.required_decel[first.frame],
                    peak.commanded_decel_mps2,
                    peak.frame,
                ),
            )
        )

    if unwarranted_states:
        first = unwarranted_states[0]
        peak = max(unwarranted_states, key=lambda r: r.commanded_decel_mps2)
        findings.append(
            Finding(
                "unwarranted_authority_state",
                "declared %s on %d frame(s) with no hazard and NO EMERGENCY BRAKING to go "
                "with it; first at frame %d (brake=%.2f = %.1f m/s^2, true gap %.1f m, true "
                "closing %+.2f m/s, true requirement %.2f m/s^2), hardest command over those "
                "frames %.2f m/s^2 -- below the %.1f m/s^2 that would make it an "
                "intervention. This is a mis-classification of the world, not a phantom "
                "brake: nothing was actuated, so no occupant felt it and no follower was "
                "endangered by it. It is still a finding, because the state is the system's "
                "declaration that it has given up on the driving task, and a system that "
                "declares that for nothing will either hand the vehicle back for nothing or "
                "start braking for nothing on the next frame. Reported separately from "
                "phantom_intervention so the two cannot be confused: they have different "
                "consequences and different fixes. Violations: %s"
                % (
                    first.safety_state.value,
                    len(unwarranted_states),
                    first.frame,
                    first.command.brake,
                    first.commanded_decel_mps2,
                    first.true.gap_m,
                    first.true.closing_mps,
                    verdict.required_decel[first.frame],
                    peak.commanded_decel_mps2,
                    truth.EMERGENCY_DECEL_MPS2,
                    ", ".join(first.violations[:4]) or "none",
                ),
            )
        )

    # -------------------------------------------- sub-emergency band --------
    if len(unwarranted) > exp.unwarranted_brake_frames_allowed:
        first = unwarranted[0]
        peak = max(unwarranted, key=lambda r: r.commanded_decel_mps2)
        findings.append(
            Finding(
                "unwarranted_brake",
                "%d frame(s) braked above the %.1f m/s^2 a headway law may use while the "
                "true requirement was zero (allowed %d); first at frame %d commanding "
                "%.2f m/s^2, peak %.2f m/s^2 at frame %d (true gap %.1f m, true closing "
                "%+.2f m/s, state=%s). This is below the %.1f m/s^2 emergency threshold, so "
                "no AEB test can see it, and it is what a phantom brake degrades into rather "
                "than disappearing: the ego was dragged from %.2f to %.2f m/s."
                % (
                    len(unwarranted),
                    truth.HEADWAY_DECEL_ALLOWANCE_MPS2,
                    exp.unwarranted_brake_frames_allowed,
                    first.frame,
                    first.commanded_decel_mps2,
                    peak.commanded_decel_mps2,
                    peak.frame,
                    first.true.gap_m,
                    first.true.closing_mps,
                    first.safety_state.value,
                    truth.EMERGENCY_DECEL_MPS2,
                    records[0].true.ego_v_mps,
                    min(r.true.ego_v_mps for r in records),
                ),
            )
        )

    # ------------------------------------------------ disproportionate ------
    if len(excessive) > exp.unjustified_brake_frames_allowed:
        first = excessive[0]
        findings.append(
            Finding(
                "disproportionate_brake",
                "%d frame(s) braked harder than the true kinematics justify (allowed %d); "
                "first at frame %d: commanded %.2f m/s^2 where %.2f m/s^2 was required and "
                "%.2f m/s^2 is the ceiling with the design margin (true gap %.1f m, true "
                "closing %+.2f m/s). Over-braking is not free: it moves the collision to the "
                "vehicle behind."
                % (
                    len(excessive),
                    exp.unjustified_brake_frames_allowed,
                    first.frame,
                    first.commanded_decel_mps2,
                    verdict.required_decel[first.frame],
                    verdict.justified_decel_mps2(first.frame),
                    first.true.gap_m,
                    first.true.closing_mps,
                ),
            )
        )

    # --------------------------------------------------------------- jerk ---
    # Two rules, both re-derived, because the old one penalised the two things a
    # correct system does: braking slightly BEFORE the oracle's causal
    # requirement becomes visible, and letting the brake off briskly afterwards.
    #
    #   1. ONLY A RISING DEMAND IS ASSESSED.  Releasing a brake is not a comfort
    #      hazard, and a ceiling on the release rate contradicts every other
    #      assertion here that demands an unwarranted deceleration be removed
    #      promptly.  See :func:`oracle.jerk_is_assessable`.
    #   2. THE BAND COMES FROM THE DEMAND, not from whether the oracle's
    #      requirement had already crossed the comfort line in the trailing
    #      window.  A demand that reaches emergency grade is a collision-
    #      avoidance action and gets the emergency ceiling whether or not it was
    #      warranted -- whether it should have existed at all is asked once, by
    #      the phantom and proportionality findings, and asking it twice through
    #      the jerk ceiling is what punished early intervention.  Below
    #      emergency grade the oracle's SYMMETRIC window still opens the
    #      emergency band, so half a second of anticipation is competence.
    #      See :func:`oracle.jerk_ceiling_mps3`.
    jerks = commanded_jerk_series(records, scenario.config)
    worst_jerk: Optional[Tuple[int, float, float]] = None
    jerk_frames = 0
    for i, jerk in enumerate(jerks):
        if i == 0 or jerk is None:
            continue
        if records[i].true.ego_v_mps <= STANDSTILL_MPS and (
            records[i - 1].true.ego_v_mps <= STANDSTILL_MPS
        ):
            # Stopped.  A brake demand that changes while the vehicle is already
            # at rest moves nobody's head.
            continue
        prev_d = records[i - 1].commanded_decel_mps2
        cur_d = records[i].commanded_decel_mps2
        if not truth.jerk_is_assessable(prev_d, cur_d):
            continue
        limit = (
            exp.max_jerk_mps3
            if exp.max_jerk_mps3 is not None
            else truth.jerk_ceiling_mps3(
                prev_d,
                cur_d,
                verdict.emergency_warranted_near(records[i].frame)
                or exempt[i]
                or blind_manoeuvre[i],
            )
        )
        if jerk > limit + 1e-6:
            jerk_frames += 1
            if worst_jerk is None or jerk - limit > worst_jerk[1] - worst_jerk[2]:
                worst_jerk = (records[i].frame, jerk, limit)
    if worst_jerk is not None:
        frame_i, jerk_v, limit_v = worst_jerk
        findings.append(
            Finding(
                "excess_jerk",
                "the demanded deceleration was INCREASED at %.1f m/s^3 at frame %d, above "
                "the %.1f m/s^3 this situation allows, on %d frame(s). %.1f m/s^3 over a "
                "%.0f ms frame is a step of %.2f m/s^2 in the demand. The ceiling outside a "
                "collision-avoidance manoeuvre is %.1f m/s^3 (the top of the band a seated "
                "occupant does not register) and inside one it is %.1f m/s^3 (full %.1f "
                "m/s^2 authority reached in the 0.4 s a human panic brake takes); braking "
                "faster than that buys no stopping distance, because the brake actuator's "
                "own rise time filters it out, and costs a head-toss the occupant cannot "
                "brace for. Only increases are counted: letting the brake off is not a "
                "comfort hazard, and this specification demands elsewhere that an "
                "unwarranted deceleration be removed promptly."
                % (
                    jerk_v,
                    frame_i,
                    limit_v,
                    jerk_frames,
                    jerk_v,
                    records[min(frame_i, len(records) - 1)].true.dt_s * 1000.0,
                    jerk_v * records[min(frame_i, len(records) - 1)].true.dt_s,
                    truth.COMFORT_JERK_MPS3,
                    truth.EMERGENCY_JERK_MPS3,
                    scenario.config.max_brake_decel_mps2,
                )
            )
        )

    # -------------------------------------- forbidden emergency authority ---
    # ``Expectation.forbid_emergency_intervention`` was declared, set on eight
    # scenarios, and READ BY NOTHING.  Those eight asserted nothing at all on
    # this axis for as long as the field existed, which is the worst kind of
    # defect a test harness can have: a green light with no lamp behind it.
    # Wired here, and split along the same line as the phantom findings, because
    # commanding an unwarranted deceleration and declaring an unwarranted state
    # are different failures.  Unlike the phantom findings this one does not ask
    # the oracle whether the frame was quiet: the scenario has ALREADY asserted,
    # in its physics, that no emergency exists anywhere in the run, so any
    # emergency-grade authority at all is a finding.  It is restricted to
    # healthy-perception frames because a perception dropout is its own reason
    # for a minimum-risk manoeuvre, and the scenarios that inject one say so.
    if exp.forbid_emergency_intervention:
        healthy = [r for r in records if r.perception_ok]
        braked = [
            r for r in healthy if r.commanded_decel_mps2 >= truth.EMERGENCY_DECEL_MPS2
        ]
        declared = [r for r in healthy if r.safety_state is SafetyState.MIN_RISK_MANEUVER]
        if braked:
            peak = max(braked, key=lambda r: r.commanded_decel_mps2)
            findings.append(
                Finding(
                    "forbidden_emergency_brake",
                    "this scenario contains no emergency at any frame, and the system "
                    "commanded emergency-grade deceleration on %d of the %d healthy frame(s); "
                    "first at frame %d (%.2f m/s^2), peak %.2f m/s^2 at frame %d (true gap "
                    "%.1f m, true closing %+.2f m/s, true requirement %.2f m/s^2). The ego "
                    "was dragged from %.2f to %.2f m/s."
                    % (
                        len(braked),
                        len(healthy),
                        braked[0].frame,
                        braked[0].commanded_decel_mps2,
                        peak.commanded_decel_mps2,
                        peak.frame,
                        braked[0].true.gap_m,
                        braked[0].true.closing_mps,
                        verdict.required_decel[braked[0].frame]
                        if braked[0].frame < len(verdict.required_decel)
                        else 0.0,
                        records[0].true.ego_v_mps,
                        min(r.true.ego_v_mps for r in records),
                    ),
                )
            )
        if declared:
            findings.append(
                Finding(
                    "forbidden_emergency_state",
                    "this scenario contains no emergency at any frame, and the system entered "
                    "%s on %d of the %d healthy frame(s); first at frame %d, where it "
                    "commanded %.2f m/s^2 (violations: %s). A minimum-risk manoeuvre is the "
                    "system's declaration that it can no longer drive; declaring it on a road "
                    "where nothing is happening is a mis-classification whether or not any "
                    "brake went with it."
                    % (
                        declared[0].safety_state.value,
                        len(declared),
                        len(healthy),
                        declared[0].frame,
                        declared[0].commanded_decel_mps2,
                        ", ".join(declared[0].violations[:4]) or "none",
                    ),
                )
            )

    # ------------------------------------------------------ pedal conflict ---
    conflicts = [r for r in records if r.pedal_conflict]
    raw_conflicts = [
        r
        for r in records
        if r.raw_command.throttle > PEDAL_CONFLICT_EPS
        and r.raw_command.brake > PEDAL_CONFLICT_EPS
    ]
    plant_conflicts = [r for r in records if getattr(r.true, "pedal_conflict", False)]
    if conflicts or raw_conflicts or plant_conflicts:
        first = (conflicts or raw_conflicts or plant_conflicts)[0]
        findings.append(
            Finding(
                "pedal_conflict",
                "throttle and brake commanded together: %d actuated frame(s), %d "
                "primary-path frame(s), %d frame(s) the plant recorded; first at frame %d "
                "(actuated throttle %.2f / brake %.2f, primary throttle %.2f / brake %.2f). "
                "There is no manoeuvre whose answer is both pedals: whichever one wins, the "
                "other was wrong, and netting them into a single acceleration demand is how "
                "the conflict stayed invisible."
                % (
                    len(conflicts),
                    len(raw_conflicts),
                    len(plant_conflicts),
                    first.frame,
                    first.command.throttle,
                    first.command.brake,
                    first.raw_command.throttle,
                    first.raw_command.brake,
                )
            )
        )

    # ------------------------------------------- the system's own findings ---
    self_reported = collect_assertable_self_reports(records, exp.allowed_self_reports)
    if self_reported:
        codes = sorted(self_reported)
        first_code = codes[0]
        first_frame = self_reported[first_code][0]
        findings.append(
            Finding(
                "self_reported_violation",
                "the system reported %d assertable finding(s) about its own output or about "
                "the motion it produced, and nothing failed: %s. First is %r at frame %d. "
                "These are the system's own words. A finding filed as advisory because the "
                "arbiter caused it is still a finding: the suffix is an attribution, not an "
                "exemption, and an occupant does not feel the attribution."
                % (
                    sum(len(v) for v in self_reported.values()),
                    ", ".join(
                        "%s x%d" % (c, len(self_reported[c])) for c in codes
                    ),
                    first_code,
                    first_frame,
                )
            )
        )

    # ------------------------------------------------------- independence ---
    if exp.primary_must_stop:
        peak_raw = max(r.raw_decel_mps2 for r in records)
        if peak_raw < truth.EMERGENCY_DECEL_MPS2:
            findings.append(
                Finding(
                    "primary_path_did_not_stop",
                    "the planner and controller together never asked for more than "
                    "%.2f m/s^2 (needed %.1f m/s^2) even though the oracle required up to "
                    "%.2f m/s^2. The arbiter is a BACKSTOP; a primary path that cannot stop "
                    "the vehicle makes the backstop load-bearing, and then a single arbiter "
                    "defect is a collision."
                    % (
                        peak_raw,
                        truth.EMERGENCY_DECEL_MPS2,
                        max(
                            (d for d in verdict.required_decel if math.isfinite(d)),
                            default=0.0,
                        ),
                    ),
                )
            )

    if exp.arbiter_must_veto_brake_below_mps2 is not None:
        limit = exp.arbiter_must_veto_brake_below_mps2
        worst = max(records, key=lambda r: r.arbiter_decel_mps2)
        if worst.arbiter_decel_mps2 > limit + 1e-6:
            findings.append(
                Finding(
                    "arbiter_failed_to_veto",
                    "the primary path demanded up to %.2f m/s^2 and the arbiter passed "
                    "%.2f m/s^2 of it through at frame %d; this scenario requires the "
                    "arbiter to hold it below %.2f m/s^2. An arbiter that cannot subtract "
                    "authority from the primary path is a wire, not an authority, and the "
                    "phantom brake it is supposed to catch originates upstream of it."
                    % (
                        max(r.raw_decel_mps2 for r in records),
                        worst.arbiter_decel_mps2,
                        worst.frame,
                        limit,
                    ),
                )
            )

    # ------------------------------------------------------------ ceiling ---
    if exp.max_commanded_decel_mps2 is not None:
        worst = max(records, key=lambda r: r.commanded_decel_mps2)
        if worst.commanded_decel_mps2 > exp.max_commanded_decel_mps2 + 1e-6:
            findings.append(
                Finding(
                    "over_braked",
                    "peak commanded deceleration %.2f m/s^2 at frame %d exceeds the %.2f m/s^2 "
                    "this scenario permits" % (
                        worst.commanded_decel_mps2,
                        worst.frame,
                        exp.max_commanded_decel_mps2,
                    ),
                )
            )

    if exp.max_safety_state is not None:
        limit = _STATE_ORDER[exp.max_safety_state]
        bad = next((r for r in records if _STATE_ORDER[r.safety_state] > limit), None)
        if bad is not None:
            findings.append(
                Finding(
                    "over_authority_state",
                    "reached %s at frame %d; this scenario permits at most %s (violations: %s)"
                    % (
                        bad.safety_state.value,
                        bad.frame,
                        exp.max_safety_state.value,
                        ", ".join(bad.violations[:4]) or "none",
                    ),
                )
            )

    disengaged = next((r for r in records if r.safety_state is SafetyState.DISENGAGE), None)
    if disengaged is not None and (
        exp.max_safety_state is None or _STATE_ORDER[exp.max_safety_state] < 3
    ):
        findings.append(
            Finding(
                "disengaged",
                "latched DISENGAGE at frame %d (violations: %s); DISENGAGE hands a moving "
                "vehicle back to nobody and is only defensible for a genuine loss of the "
                "decision path" % (disengaged.frame, ", ".join(disengaged.violations[:4]) or "none"),
            )
        )

    # ----------------------------------------------------------- recovery ---
    if exp.recover_within_frames is not None:
        clear = _disturbance_clear_frame(records, verdict)
        recovered = _first_stable_nominal(records, clear)
        if recovered is None:
            findings.append(
                Finding(
                    "failed_to_recover",
                    "%s but the system never returned to a stable NOMINAL; it ended in %s "
                    "(violations: %s)"
                    % (
                        "nothing in this run ever warranted a degraded state"
                        if clear == 0
                        else "every disturbance was over by frame %d" % clear,
                        records[-1].safety_state.value,
                        ", ".join(records[-1].violations[:4]) or "none",
                    ),
                )
            )
        elif recovered - clear > exp.recover_within_frames:
            findings.append(
                Finding(
                    "slow_recovery",
                    "returned to NOMINAL at frame %d, %d frames (%.2f s) after the last "
                    "disturbance cleared at frame %d; the limit is %d frames"
                    % (
                        recovered,
                        recovered - clear,
                        (recovered - clear) * scenario.config.dt_s,
                        clear,
                        exp.recover_within_frames,
                    ),
                )
            )

    # -------------------------------------------------------------- speed ---
    if exp.min_speed_floor_mps is not None:
        slowest = min(records, key=lambda r: r.true.ego_v_mps)
        if slowest.true.ego_v_mps < exp.min_speed_floor_mps:
            findings.append(
                Finding(
                    "speed_loss",
                    "ego slowed to %.2f m/s at frame %d; nothing in the true world justified "
                    "dropping below %.2f m/s (true gap there %.1f m, true closing %+.2f m/s)"
                    % (
                        slowest.true.ego_v_mps,
                        slowest.frame,
                        exp.min_speed_floor_mps,
                        slowest.true.gap_m,
                        slowest.true.closing_mps,
                    ),
                )
            )

    if exp.min_final_speed_mps is not None:
        final = records[-1].true.ego_v_mps
        if final < exp.min_final_speed_mps:
            findings.append(
                Finding(
                    "final_speed",
                    "ego finished at %.2f m/s; at least %.2f m/s was required"
                    % (final, exp.min_final_speed_mps),
                )
            )

    if exp.max_final_speed_mps is not None:
        final = records[-1].true.ego_v_mps
        if final > exp.max_final_speed_mps:
            findings.append(
                Finding(
                    "final_speed",
                    "ego finished at %.2f m/s; this scenario requires it to be brought below "
                    "%.2f m/s" % (final, exp.max_final_speed_mps),
                )
            )

    # ------------------------------------------------------------ lateral ---
    if exp.max_abs_lateral_offset_m is not None:
        worst = max(records, key=lambda r: abs(r.true.lateral_offset_m))
        if abs(worst.true.lateral_offset_m) > exp.max_abs_lateral_offset_m:
            findings.append(
                Finding(
                    "lane_departure",
                    "ego reached %.2f m from the lane centre at frame %d; the limit is %.2f m"
                    % (worst.true.lateral_offset_m, worst.frame, exp.max_abs_lateral_offset_m),
                )
            )

    # --------------------------------------------------------------- exit ---
    if exp.require_fail_safe_exit:
        if not exit_commands:
            findings.append(
                Finding("no_exit_command", "the run ended without writing any command")
            )
        else:
            last_in_loop = records[-1]
            throttling = next((c for c in exit_commands if c.throttle > 0.0), None)
            if throttling is not None:
                findings.append(
                    Finding(
                        "exit_throttle",
                        "an exit command still applies throttle %.2f; leaving the loop must "
                        "release it" % throttling.throttle,
                    )
                )
            if last_in_loop.command.brake > 0.05 and exit_commands[-1].brake < last_in_loop.command.brake - 1e-6:
                findings.append(
                    Finding(
                        "exit_released_brake",
                        "the loop ended mid-intervention with brake %.2f but the exit command "
                        "brakes only %.2f" % (last_in_loop.command.brake, exit_commands[-1].brake),
                    )
                )
            if last_in_loop.true.ego_v_mps > 0.5 and max(c.brake for c in exit_commands) <= 0.0:
                findings.append(
                    Finding(
                        "exit_no_brake",
                        "the run ended at %.2f m/s and no exit command applies any brake"
                        % last_in_loop.true.ego_v_mps,
                    )
                )

    return findings


# --------------------------------------------------------------------------- #
# Independence: the same physics, one layer at a time
# --------------------------------------------------------------------------- #
#
# These live here rather than in ``library`` because they are scenarios ABOUT
# the stack decomposition, which is the thing this module defines.  They are
# picked up by ``report.all_scenarios()`` and appear in the table, the JSON
# artifact and the baseline exactly like every other case.

_INDEPENDENCE_WORLD = (
    "A stopped vehicle 40 m ahead, ego closing at 20 m/s on a straight dry road. "
    "Keeping the 2.0 m standstill clearance leaves 38 m of usable closure, so the "
    "constant deceleration required is 20^2 / (2 x 38) = 5.26 m/s^2 -- well above "
    "the 3.0 m/s^2 comfort line, unmistakably an emergency, and two thirds of the "
    "8.0 m/s^2 the vehicle has. Full braking from 20 m/s covers 25.0 m plus about "
    "1.3 m for the 0.15 s brake rise, and one frame of sense latency adds 1.0 m, so "
    "27.3 m of the 40 m is enough: contact here is not a hard case, it is a system "
    "that did not brake. "
)

_INDEPENDENCE_STATIONARY = LeadSpec(40.0, 0.0, lead_stationary(), label="stopped car")

DEGRADED_STACK_SCENARIOS: Tuple[Scenario, ...] = (
    Scenario(
        name="arbiter_alone_stops_for_stationary",
        summary="planner blinded; only the arbiter can see the stopped car",
        guards="arbiter independence -- the backstop must not need the primary path",
        physics=_INDEPENDENCE_WORLD
        + "The planner is handed an EMPTY object list every frame while perception "
        "stays healthy and the arbiter still receives the real tracks. This is the "
        "arbiter's entire reason to exist: it is advertised as an INDEPENDENT "
        "backstop, and a backstop that only works when the thing it is backing up "
        "also works is not one. The failure being tested for is structural, not "
        "numeric -- if every brake in this stack ultimately originates in the "
        "planner's object list, then one dropped list is one collision, and no "
        "amount of tuning inside the arbiter changes that. REQUIRED: emergency-grade "
        "braking before the last avoidance frame, and no contact.",
        frames=140,
        ego_speed_mps=20.0,
        lead=_INDEPENDENCE_STATIONARY,
        stack=ARBITER_ONLY_STACK,
        expect=Expectation(
            must_intervene=True,
            min_clearance_m=0.5,
            max_abs_lateral_offset_m=0.5,
        ),
    ),
    Scenario(
        name="primary_alone_stops_for_stationary",
        summary="arbiter stripped of authority; the primary path must stop by itself",
        guards="primary-path competence -- the backstop must not be load-bearing",
        physics=_INDEPENDENCE_WORLD
        + "Here the whole stack sees the car but the ARBITER'S command is discarded "
        "and the planner-plus-controller command is actuated. This is the converse "
        "claim and it is just as load-bearing: if the primary path cannot stop for a "
        "stopped car, then every stop this system has ever made was the arbiter's, "
        "the arbiter is not a backstop but the driver, and its single-point failures "
        "are the vehicle's. It also explains the oscillation this module is being "
        "redesigned out of: when the only brake in the system is the safety monitor, "
        "every tuning change has to trade phantom braking against missed braking, "
        "because there is nothing else to carry the ordinary case. REQUIRED: the "
        "primary path reaches emergency-grade deceleration on its own, and no contact.",
        frames=140,
        ego_speed_mps=20.0,
        lead=_INDEPENDENCE_STATIONARY,
        stack=PRIMARY_ONLY_STACK,
        expect=Expectation(
            must_intervene=True,
            primary_must_stop=True,
            min_clearance_m=0.5,
            max_abs_lateral_offset_m=0.5,
        ),
    ),
    Scenario(
        name="arbiter_vetoes_stuck_brake_on_empty_road",
        summary="primary path jams full brake on an empty motorway; the arbiter must veto",
        guards="round-1 phantom AEB originating UPSTREAM of the arbiter",
        physics="Empty road, no object anywhere in the world, ego cruising at 25 m/s, and "
        "the primary path emits brake=1.00 on every frame. The true collision-avoidance "
        "requirement is identically zero for the whole run, so 8.0 m/s^2 here is not a "
        "mis-tuned response to a hazard, it is a response to nothing. The consequence is "
        "physical: a follower keeping a 2 s gap and taking 1 s to react can absorb about "
        "5 m/s^2 of lead deceleration and cannot absorb 8, so an unwarranted full-authority "
        "stop on a motorway does not avoid a collision, it manufactures one behind. The "
        "arbiter holds authority over the longitudinal channel and is the last component "
        "before the actuators, so it is the only place this can be bounded. REQUIRED: the "
        "arbiter's own command stays under the 3.0 m/s^2 comfort line -- it must SUBTRACT "
        "authority, not merely refrain from adding it. An arbiter that can only ever brake "
        "harder than the plan is a wire with logging, and the phantom braking measured on "
        "real video originates upstream of it, where it structurally cannot be vetoed.",
        frames=120,
        ego_speed_mps=25.0,
        lead=None,
        stack=StackSpec(controller=CONTROLLER_STUCK_BRAKE),
        expect=Expectation(
            forbid_emergency_intervention=True,
            arbiter_must_veto_brake_below_mps2=truth.COMFORT_DECEL_MPS2,
            min_speed_floor_mps=20.0,
        ),
    ),
    Scenario(
        name="arbiter_alone_stops_runaway_throttle",
        summary="primary path floors the throttle at a stopped car; only the arbiter is left",
        guards="arbiter independence against a commanding, not merely absent, primary path",
        physics=_INDEPENDENCE_WORLD
        + "The primary path emits throttle=1.00 every frame -- a wound-up integrator, a "
        "jammed output stage, a planner that has concluded the road is clear. This is "
        "strictly harder than the blinded-planner case and it is the case that decides "
        "whether the arbiter has AUTHORITY or merely a veto on its own additions: the ego "
        "is being accelerated toward the obstacle at 2.5 m/s^2 while the arbiter is deciding. "
        "The obstacle is real, it is in the arbiter's own track list, and the kinematics are "
        "not marginal. If the vehicle reaches it, the independent backstop is a label. "
        "REQUIRED: emergency-grade braking, no contact, and no throttle surviving to the "
        "actuators while the brake is applied.",
        frames=160,
        ego_speed_mps=20.0,
        lead=_INDEPENDENCE_STATIONARY,
        stack=StackSpec(controller=CONTROLLER_STUCK_THROTTLE),
        expect=Expectation(
            must_intervene=True,
            min_clearance_m=0.5,
            max_abs_lateral_offset_m=0.5,
        ),
    ),
)
"""Scenarios that hold ONE layer of the longitudinal path to the whole requirement.

Two of them make the same physical demand -- stop for a stopped car 40 m away --
of two disjoint halves of the stack, and both must satisfy it.  The other two
present each half with a demand the other half must overrule.
"""


def expectation_field_reads() -> Dict[str, int]:
    """Every :class:`Expectation` field, and how many times ``evaluate`` reads it.

    A field with a count of zero is an assertion that is DECLARED, SET ON REAL
    SCENARIOS, AND NEVER CHECKED -- a green light with no lamp behind it, and the
    worst defect a test harness can have, because the scenarios carrying it
    report success on an axis nothing measured.
    ``forbid_emergency_intervention`` was in that state on eight scenarios.

    The check is by source inspection of :func:`evaluate` rather than by a
    hand-maintained registry, on purpose: a registry is another thing to forget
    to update, and the failure mode being guarded against is precisely
    forgetting.
    """
    import inspect

    body = inspect.getsource(evaluate)
    return {
        f.name: body.count("exp." + f.name) for f in dataclass_fields(Expectation)
    }


def unread_expectation_fields() -> List[str]:
    """Expectation fields ``evaluate`` never reads.  Must be empty."""
    return sorted(name for name, hits in expectation_field_reads().items() if hits == 0)


def commanded_jerk_series(
    records: Sequence[FrameRecord], config: PlantConfig = DEFAULT_PLANT
) -> List[Optional[float]]:
    """Per-frame magnitude of the rate of change of the DEMANDED deceleration.

    ``out[i]`` is ``|decel[i] - decel[i-1]| / dt`` in m/s^3, where ``dt`` is the
    real duration of the step that separated the two commands -- not the nominal
    period, because a 187 ms frame really did hold the previous command for
    187 ms and the jerk when the next one lands is correspondingly smaller.
    ``out[0]`` is None.

    The DEMAND is judged rather than the achieved acceleration.  The brake
    actuator's 0.15 s rise time smooths a step demand into something the
    accelerometer barely notices, but that lag is a property of the plumbing and
    not a safety feature: the same demand on a vehicle with a faster brake, or on
    a brake-by-wire axle, arrives in full.  A specification that credits the
    system for its actuator's sluggishness has stopped specifying the system.
    The achieved jerk is reported alongside as a metric, and the arbiter's own
    measurement of it is caught by :data:`ASSERTABLE_SELF_REPORTS`.
    """
    out: List[Optional[float]] = [None] * len(records)
    for i in range(1, len(records)):
        dt = records[i].true.dt_s
        if not dt or not math.isfinite(dt) or dt <= 0.0:
            dt = config.dt_s
        out[i] = abs(records[i].commanded_decel_mps2 - records[i - 1].commanded_decel_mps2) / dt
    return out


def achieved_jerk_series(
    records: Sequence[FrameRecord], config: PlantConfig = DEFAULT_PLANT
) -> List[Optional[float]]:
    """Per-frame magnitude of the rate of change of the TRUE ego acceleration.

    What an accelerometer bolted to the seat rail would integrate.  Reported as
    a metric rather than asserted on, because the assertion belongs on the
    demand -- see :func:`commanded_jerk_series`.
    """
    out: List[Optional[float]] = [None] * len(records)
    for i in range(1, len(records)):
        dt = records[i].true.dt_s
        if not dt or not math.isfinite(dt) or dt <= 0.0:
            dt = config.dt_s
        out[i] = abs(records[i].true.ego_a_mps2 - records[i - 1].true.ego_a_mps2) / dt
    return out


def is_assertable_self_report(code: str) -> bool:
    """Whether one ``ArbitrationResult.violations`` entry is a finding here.

    Prefix match against :data:`ASSERTABLE_SELF_REPORTS`.  See that constant for
    the rule, and for why the ``_arbiter_induced`` and ``_arbiter_commanded``
    suffixes do not exempt anything.
    """
    return any(code.startswith(prefix) for prefix in ASSERTABLE_SELF_REPORTS)


def self_report_family(code: str) -> str:
    """The stable family name of a self-report, with the numbers stripped.

    ``jerk_32.2_above_15.0_arbiter_induced`` -> ``jerk_above_limit``.  Findings
    carry measured values in their text, which makes them useless as dictionary
    keys and unreadable in a diff of two report artifacts; the family is what a
    baseline and a regression comparison need.
    """
    for prefix, family in (
        ("jerk_", "jerk_above_limit"),
        ("measured_accel_", "measured_accel_above_limit"),
        ("measured_decel_", "measured_decel_above_limit"),
        ("command_throttle_out_of_range", "command_throttle_out_of_range"),
        ("command_brake_out_of_range", "command_brake_out_of_range"),
        ("command_steering_out_of_range", "command_steering_out_of_range"),
    ):
        if code.startswith(prefix):
            return family
    return code


def collect_assertable_self_reports(
    records: Sequence[FrameRecord], allowed: Sequence[str] = ()
) -> Dict[str, List[int]]:
    """Map assertable self-report family -> the frames it was reported on.

    Args:
        records: The run.
        allowed: Prefixes the scenario has excused; see
            :attr:`Expectation.allowed_self_reports`.

    Returns:
        A dict, empty when the system reported nothing assertable about itself.
    """
    out: Dict[str, List[int]] = {}
    for r in records:
        for code in r.violations:
            if not is_assertable_self_report(code):
                continue
            if any(code.startswith(prefix) for prefix in allowed):
                continue
            out.setdefault(self_report_family(code), []).append(r.frame)
    return out


def _manoeuvre_exemptions(
    records: Sequence[FrameRecord], verdict: truth.OracleVerdict
) -> Tuple[List[bool], List[bool]]:
    """Per frame: ``(inside a warranted stop, inside a blind manoeuvre)``.

    Two different exemptions, kept apart because they mean different things and
    conflating them turns one into a bug.

    ``warranted``
        The frame belongs to an unbroken braking run that WAS kinematically
        warranted when it began.  A correct intervention destroys the evidence
        for itself -- once the brake has worked the gap stops shrinking and the
        requirement falls to zero -- so the last frames of every successful stop
        look exactly like a phantom.  These frames are still judged for
        PROPORTIONALITY, against the worst requirement in the trailing window,
        because the requirement they are answering is a real one.

    ``blind``
        The frame belongs to a braking run that began, or continued, while
        PERCEPTION WAS DOWN.  A controlled stop under sensor loss is a manoeuvre
        this specification demands (``source_loss_mid_run`` requires the vehicle
        to be stopped by the end of it), and it is warranted by the system's
        blindness rather than by anything in the oracle's kinematics -- which
        see an empty road.  There is no kinematic requirement to be proportional
        TO, so these frames are exempt from the proportionality tests outright.
        Judging them against the 1.5 m/s^2 a headway law may use would make
        every minimum-risk stop a finding, and the release tail of one a finding
        for four more frames after perception came back.
    """
    warranted = [False] * len(records)
    blind = [False] * len(records)
    run_warranted = False
    run_blind = False
    for i, r in enumerate(records):
        if r.command.brake > 0.05 or not r.perception_ok:
            if verdict.warranted_at(r.frame):
                run_warranted = True
            if not r.perception_ok:
                run_blind = True
        else:
            run_warranted = False
            run_blind = False
        warranted[i] = run_warranted
        blind[i] = run_blind
    return warranted, blind


def _completing_a_warranted_stop(
    records: Sequence[FrameRecord], verdict: truth.OracleVerdict
) -> List[bool]:
    """Per frame: is the system still inside a braking run that WAS warranted?

    A correct intervention destroys the evidence for itself -- once the brake
    has worked, the gap stops shrinking and the kinematic requirement falls to
    zero -- so the last frames of every successful stop look exactly like a
    phantom.  The distinguishing feature is continuity: a phantom brake begins
    with no warrant, while the tail of a real one belongs to an unbroken run of
    braking that started when there was one.

    A run ends when the command releases the brake below 0.05 (0.4 m/s^2, which
    is inside the noise of a jerk-limited release).

    A braking run that began while PERCEPTION WAS DOWN counts as warranted too.
    A controlled stop under sensor loss is a manoeuvre the specification demands
    (``source_loss_mid_run`` requires the vehicle to be stopped by the end of
    it), and it is warranted by the system's blindness rather than by anything
    in the oracle's kinematics -- which see an empty road and would score every
    frame of it as a phantom.
    """
    out = [False] * len(records)
    run_start: Optional[int] = None
    run_warranted = False
    for i, r in enumerate(records):
        if r.command.brake > 0.05:
            if run_start is None:
                run_start = i
                run_warranted = False
            if verdict.warranted_at(r.frame) or not r.perception_ok:
                run_warranted = True
        else:
            run_start = None
            run_warranted = False
        out[i] = run_warranted
    return out


def _disturbance_clear_frame(
    records: Sequence[FrameRecord], verdict: truth.OracleVerdict
) -> int:
    """The first frame from which nothing external justifies any restriction.

    Recovery cannot be measured from the end of the kinematic hazard alone: a
    perception dropout or a detection miss is also a legitimate reason to hold a
    degraded state.  The baseline is therefore the latest of

    * the oracle's ``hazard_clear_frame``,
    * one frame after the last unhealthy perception frame,
    * one frame after the last frame on which a present lead went undetected.

    From that frame on, everything the system can observe is benign, so a
    persistent degraded state is the system sustaining itself.
    """
    baseline = verdict.hazard_clear_frame or 0
    was_present = records[0].true.lead_present
    for r in records:
        if not r.perception_ok:
            baseline = max(baseline, r.frame + 1)
        elif r.true.lead_present and not r.detected:
            baseline = max(baseline, r.frame + 1)
        if r.true.lead_present and not was_present:
            # A vehicle entering the lane is a new situation to assess, so the
            # clock on returning to NOMINAL starts again from the cut-in.
            baseline = max(baseline, r.frame)
        was_present = r.true.lead_present
    return min(baseline, records[-1].frame)


def _first_stable_nominal(records: Sequence[FrameRecord], from_frame: int) -> Optional[int]:
    """First frame at or after ``from_frame`` from which the state stays NOMINAL.

    "Stable" matters: a system oscillating in and out of MIN_RISK_MANEUVER has
    not recovered, and one that touches NOMINAL for a single frame before
    re-latching would otherwise pass.
    """
    tail_ok = True
    answer: Optional[int] = None
    for r in reversed(records):
        if r.frame < from_frame:
            break
        if r.safety_state is SafetyState.NOMINAL and tail_ok:
            answer = r.frame
        else:
            tail_ok = False
    return answer
