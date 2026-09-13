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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from adas.core.models import (
    ArbitrationResult,
    ControlCommand,
    EgoState,
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


# --------------------------------------------------------------------------- #
# The system under test
# --------------------------------------------------------------------------- #


@dataclass
class FrameRecord:
    """One frame of the closed loop: truth in, command out."""

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

    @property
    def commanded_decel_mps2(self) -> float:
        """The deceleration the actuated command asks for, m/s^2."""
        return self.command.brake * DEFAULT_PLANT.max_brake_decel_mps2


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

    def __init__(self, cruise_speed_mps: float) -> None:
        self.planner = BehaviorPlanner(
            cruise_speed_mps=cruise_speed_mps,
            ego_lane_half_width_frac=PLANNER_EGO_LANE_HALF_WIDTH_FRAC,
        )
        self.controller = PIDLikeLongitudinalController()
        self.monitor = SafetyMonitor()
        self._last_speed_mps = cruise_speed_mps

    def step(self, obs: Observation, dt_s: float) -> Tuple[ControlCommand, ControlCommand, object]:
        """Run one frame.

        Returns:
            ``(actuated_command, raw_controller_command, arbitration_result)``.
        """
        self._last_speed_mps = obs.ego.speed_mps
        try:
            plan = self.planner.plan(
                frame_width_px=obs.frame_width_px,
                lane_center_px=obs.lane_center_px,
                objects=obs.tracks,
                ego=obs.ego,
                perception_valid=obs.perception.ok,
                dt_s=dt_s,
                lane=obs.lane,
                frame_height_px=obs.frame_height_px,
            )
            emergency = "aeb" in (plan.reason or "") or "emergency" in (plan.reason or "")
            raw = self.controller.to_command(
                plan, obs.ego.speed_mps, dt_s=dt_s, emergency=emergency
            )
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
            return fail, ControlCommand(0.0, 0.0, 0.0), result

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
        return result.command, raw, result

    def fail_safe(self, t_s: float, ego_speed_mps: float, dt_s: float) -> ControlCommand:
        """The command to actuate when the decision path is gone, or at exit.

        The arbiter is run with no plan, a neutral command and a failed
        perception status.  Per its own contract that puts it into a
        minimum-risk manoeuvre and returns its rate-shaped braking command.
        """
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
        return {
            "frames": len(self.records),
            "min_gap_m": _round(self.verdict.min_gap_m),
            "collided": self.verdict.collided,
            "oracle_emergency": self.verdict.emergency,
            "oracle_first_emergency_frame": self.verdict.first_emergency_frame,
            "oracle_last_avoidance_frame": self.verdict.last_avoidance_frame,
            "oracle_hazard_clear_frame": self.verdict.hazard_clear_frame,
            "oracle_avoidable": self.verdict.avoidable,
            "max_commanded_decel_mps2": _round(max(decels) if decels else 0.0),
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
        )
    )

    history: List[WorldState] = []
    records: List[FrameRecord] = []
    state = plant.state
    for _ in range(scenario.frames):
        history.append(state)
        obs = sensor.observe(state)
        command, raw, result = system.step(obs, dt)
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
            findings.append(
                Finding(
                    "missed_intervention",
                    "the oracle required %.2f m/s^2 from frame %s and the system never "
                    "commanded %.1f m/s^2 (peak %.2f m/s^2)"
                    % (
                        max(d for d in verdict.required_decel if math.isfinite(d)),
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
    excessive: List[FrameRecord] = []
    for r in records:
        if r.true.ego_v_mps <= STANDSTILL_MPS or not r.perception_ok:
            continue
        quiet = verdict.is_quiet(r.frame) and not exempt[r.frame]
        emergency_authority = (
            r.commanded_decel_mps2 >= truth.EMERGENCY_DECEL_MPS2
            or r.safety_state is SafetyState.MIN_RISK_MANEUVER
        )
        if quiet and emergency_authority:
            phantoms.append(r)
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
