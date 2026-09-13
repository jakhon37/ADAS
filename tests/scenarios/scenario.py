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
from dataclasses import dataclass, field
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
    PERFECT_PERCEPTION,
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
        lead: The lead vehicle script, or None for an empty road.
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
    )
    sensor = Sensor(scenario.perception)
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

    exit_commands: List[ControlCommand] = []
    if scenario.terminate_during_run:
        exit_commands = system.shutdown(state.t_s, dt)

    verdict = truth.judge(history, scenario.lead, scenario.road, cfg)
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
    if exp.no_collision and verdict.collided:
        if verdict.avoidable is False:
            findings.append(
                Finding(
                    "collided_unavoidable",
                    "contact at frame %d; the oracle says it was already unavoidable at the "
                    "first frame the lead was visible, so this scenario is mis-specified"
                    % verdict.collision_frame,
                )
            )
        else:
            findings.append(
                Finding(
                    "collided",
                    "contact at frame %d (t=%.2f s); full-authority braking was still "
                    "sufficient up to frame %s"
                    % (
                        verdict.collision_frame,
                        records[min(verdict.collision_frame, len(records) - 1)].t_s,
                        verdict.last_avoidance_frame,
                    ),
                )
            )

    if exp.min_clearance_m is not None and verdict.min_gap_m < exp.min_clearance_m:
        findings.append(
            Finding(
                "clearance",
                "minimum true gap %.2f m, required at least %.2f m"
                % (verdict.min_gap_m, exp.min_clearance_m),
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
                        "%.1f m/s^2 (peak %.2f m/s^2)"
                        % (truth.NEGLIGIBLE_DECEL_MPS2, peak_cmd),
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
                    ),
                )
            )
        elif (
            verdict.last_avoidance_frame is not None
            and first_emergency_cmd > verdict.last_avoidance_frame
        ):
            findings.append(
                Finding(
                    "late_intervention",
                    "first emergency-grade command at frame %d, %d frames (%.2f s) after the "
                    "last frame from which full braking still avoided contact (frame %d)"
                    % (
                        first_emergency_cmd,
                        first_emergency_cmd - verdict.last_avoidance_frame,
                        (first_emergency_cmd - verdict.last_avoidance_frame) * scenario.config.dt_s,
                        verdict.last_avoidance_frame,
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
    exempt = _completing_a_warranted_stop(records, verdict)
    phantoms: List[FrameRecord] = []
    unwarranted: List[FrameRecord] = []
    excessive: List[FrameRecord] = []
    for i, r in enumerate(records):
        if r.true.ego_v_mps <= STANDSTILL_MPS or not r.perception_ok:
            continue
        quiet = verdict.is_quiet(r.frame) and not exempt[i]
        emergency_authority = (
            r.commanded_decel_mps2 >= truth.EMERGENCY_DECEL_MPS2
            or r.safety_state is SafetyState.MIN_RISK_MANEUVER
        )
        if quiet:
            if emergency_authority:
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
        findings.append(
            Finding(
                "phantom_intervention",
                "emergency authority on %d frame(s) with no hazard; first at frame %d "
                "(state=%s, brake=%.2f = %.1f m/s^2, true gap %.1f m, true closing %+.2f m/s, "
                "true requirement %.2f m/s^2)"
                % (
                    len(phantoms),
                    first.frame,
                    first.safety_state.value,
                    first.command.brake,
                    first.commanded_decel_mps2,
                    first.true.gap_m,
                    first.true.closing_mps,
                    verdict.required_decel[first.frame],
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
        limit = (
            exp.max_jerk_mps3
            if exp.max_jerk_mps3 is not None
            else truth.jerk_limit_mps3(
                verdict.emergency_warranted_at(records[i].frame) or exempt[i]
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
                "the demanded deceleration changed at %.1f m/s^3 at frame %d, above the "
                "%.1f m/s^3 this situation allows, on %d frame(s). %.1f m/s^3 over a %.0f ms "
                "frame is a step of %.2f m/s^2 in the demand. The ceiling outside an "
                "emergency is %.1f m/s^3 (the top of the band a seated occupant does not "
                "register) and inside one it is %.1f m/s^3 (full %.1f m/s^2 authority "
                "reached in the 0.4 s a human panic brake takes); braking faster than that "
                "buys no stopping distance, because the brake actuator's own rise time "
                "filters it out, and costs a head-toss the occupant cannot brace for."
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
    """
    out = [False] * len(records)
    run_start: Optional[int] = None
    run_warranted = False
    for i, r in enumerate(records):
        if r.command.brake > 0.05:
            if run_start is None:
                run_start = i
                run_warranted = False
            if verdict.warranted_at(r.frame):
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
