"""Truth oracle: what SHOULD have happened, from kinematics alone.

The oracle never looks at the system under test.  It is handed the TRUE state
history produced by :mod:`tests.scenarios.plant` and answers four questions:

1. **Was there ever a genuine emergency, and from which frame?**
2. **Was a collision avoidable, and what was the last frame from which
   full-authority braking still avoided it?**
3. **Was a given commanded deceleration justified** by the true range and the
   true closing rate at that moment?
4. **When did the hazard end?** -- so that failure to recover can be measured.

Margin conventions, stated explicitly because every judgement below depends on
them:

* :data:`CONTACT_GAP_M` = 0.0 m.  The gap is bumper to bumper, so zero is
  contact.  "Avoidable" is judged against contact, not against a comfort
  margin: a collision the vehicle could physically have avoided is a failure
  even if avoiding it would have been uncomfortable.
* :data:`REQUIRED_CLEARANCE_M` = 2.0 m.  The clearance a *correct* intervention
  is expected to preserve.  ``required_decel`` is the deceleration needed to
  stop with this much room left, not to stop just short of contact.  2.0 m is
  the standstill clearance a driver leaves and is the smallest gap at which a
  camera-based range estimate is still meaningful.
* :data:`COMFORT_DECEL_MPS2` = 3.0 m/s^2.  Above this a passenger notices, and
  an ordinary adaptive-cruise law would not go there for headway keeping.  A
  demand above comfort is therefore *collision avoidance* and must be justified
  by kinematics.
* :data:`NEGLIGIBLE_DECEL_MPS2` = 1.0 m/s^2.  Below this the situation is not a
  hazard at all.  Emergency-grade authority applied while the true requirement
  is below this floor is a PHANTOM intervention.
* :data:`HEADWAY_DECEL_ALLOWANCE_MPS2` = 1.5 m/s^2.  The authority an ordinary
  headway-keeping law may use with no kinematic hazard whatever.  It is what
  makes the SUB-EMERGENCY band measurable: braking above it with a true
  requirement of zero is unwarranted even though it never reaches AEB grade.
* :data:`COMFORT_JERK_MPS3` = 2.5 m/s^3 and :data:`EMERGENCY_JERK_MPS3` =
  20.0 m/s^3.  The rate at which the demanded deceleration may change, outside
  and inside a genuine emergency.

None of these numbers is read from :mod:`adas.control.arbiter`.  Every one of
them is derived below from occupant tolerance, from the vehicle, or from the
kinematics, and the derivation is in the docstring of the constant.

Two different assumptions about the lead appear below, and conflating them is
the classic way to build a harness that demands clairvoyance:

* ``required_decel`` uses the **causal** assumption: the lead holds its
  *current* acceleration until it stops.  That is information a real system can
  measure this frame.  This drives justification and phantom detection.
* ``last_avoidance_frame`` uses the **true future script**, because the question
  it answers ("could this have been avoided?") is a physical one about what
  actually happened, not about what was knowable.  It is only ever used to
  bound how LATE an intervention was, never to require an earlier one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from tests.scenarios.plant import (
    DEFAULT_PLANT,
    LeadSpec,
    Plant,
    PlantConfig,
    RoadSpec,
    WorldState,
)

CONTACT_GAP_M = 0.0
"""Bumper-to-bumper gap at which contact occurs, metres."""

REQUIRED_CLEARANCE_M = 2.0
"""Clearance a correct intervention should preserve, metres."""

COMFORT_DECEL_MPS2 = 3.0
"""Above this, braking is collision avoidance rather than headway keeping."""

EMERGENCY_DECEL_MPS2 = 3.5
"""Deceleration at or above which a command counts as an emergency intervention.

Half a metre above comfort, so that a comfort-limited ACC ramp cannot be
mistaken for an AEB event by rounding.
"""

NEGLIGIBLE_DECEL_MPS2 = 1.0
"""Below this true requirement there is no hazard worth the name."""

HEADWAY_DECEL_ALLOWANCE_MPS2 = 1.5
"""Deceleration a headway-keeping law may use when NOTHING is kinematically required.

Derivation, from the behaviour a following law has to produce and not from any
constant in the system under test.  The worst headway deficit an adaptive
cruise has to close without there being a collision problem is roughly one
second of time gap -- following at 1 s and wanting 2 s.  At 20 m/s that is 20 m
of gap to open, and a gap only opens while the ego is slower than the lead.
Running 2 m/s below the lead opens 20 m in 10 s, which is the slow, unobtrusive
correction a passenger should never notice; reaching that 2 m/s offset over
about 2 s costs 1.0 m/s^2.  Half a metre per second squared on top covers the
transient at the start of the correction and the discretisation of a
jerk-limited brake, giving 1.5 m/s^2 -- half of :data:`COMFORT_DECEL_MPS2`, so
it is by construction below the level at which a passenger registers braking at
all.

Above this figure, with a true requirement of zero, the system is braking for
something that is not in the world.  That is the whole SUB-EMERGENCY band: it
never reaches :data:`EMERGENCY_DECEL_MPS2`, so an AEB-grade test cannot see it,
and it is exactly the band in which a phantom that merely drags the vehicle
down hides.
"""

COMFORT_JERK_MPS3 = 2.5
"""Largest rate of change of demanded deceleration outside an emergency, m/s^3.

Human tolerance, not a control constant.  Longitudinal jerk becomes perceptible
to a seated occupant at roughly 2 m/s^3 and is the quantity that throws an
unrestrained head, a standing passenger or a loose object forward; lift and
rail ride-comfort practice caps jerk at about 2.0 m/s^3 for exactly this
reason.  2.5 m/s^3 is the top of that band.

The controllability half of the argument matters as much as the comfort half: a
driver with hands on the wheel has to be able to interpret what the automation
is doing.  A demand that changes faster than this reads as a fault rather than
as a manoeuvre and provokes exactly the wrong reaction -- an override, or a
swerve.  Outside an emergency nothing is bought by going faster, because there
is by definition no collision to outrun.
"""

EMERGENCY_JERK_MPS3 = 20.0
"""Largest rate of change of demanded deceleration during a genuine emergency.

Derivation: full authority must be reachable at least as fast as a competent
human panic brake, and there is nothing to be gained by being faster.  Measured
pedal-force rise times in emergency braking studies run 0.2-0.4 s from first
contact to maximum.  Taking the slower end, reaching the vehicle's full
8.0 m/s^2 in 0.4 s is 20 m/s^3.

Going faster does not shorten the stop.  The brake actuator's own rise time is
0.15 s (``PlantConfig.brake_rise_time_s``), so a demand that steps from zero to
full in one 50 ms frame -- 160 m/s^3 -- is filtered by the hydraulics into
almost exactly the same deceleration profile as a 20 m/s^3 ramp, while the part
that does get through is a head-toss the occupant cannot brace for and the
following driver cannot read.  A step demand also guarantees that any
mis-classification is delivered at full severity before it can be withdrawn,
which is precisely how a one-frame false positive became a 42-frame
minimum-risk manoeuvre.

Note that this ceiling is deliberately MORE permissive than the arbiter's own
jerk constant.  It is derived here from the vehicle and the occupant; if the
system's internal limit is tighter, that is the system's choice and this
specification does not contradict it.
"""

JUSTIFICATION_TOLERANCE_MPS2 = 0.5
"""Slack allowed between the commanded and the truly required deceleration.

A controller that brakes slightly harder than the minimum is being prudent, not
wrong.  Half a metre per second squared is roughly the discretisation of a
brake command that has been through a jerk limiter.
"""

JUSTIFICATION_MARGIN_FACTOR = 1.5
"""Multiplier applied to the true requirement before calling a brake excessive.

``required_decel`` is the *theoretical minimum*: it assumes the range is known
exactly and the deceleration appears instantly.  A designer must allow for
neither being true.  Fifty per cent covers a 20% range under-estimate (which, since the requirement
goes as ``v^2 / 2d``, is 25% of extra deceleration) together with the 0.15 s
brake rise and a frame of latency, so a system braking within that band is
competent rather than excessive.  Above it the demand is not explained by the
kinematics, and disproportionate braking is not free: a follower keeping its own
2 s gap and taking 1 s to react can absorb a 5 m/s^2 lead deceleration and
cannot absorb 8 m/s^2, so over-braking transfers the collision to the vehicle
behind.
"""

JUSTIFICATION_WINDOW_FRAMES = 10
"""How far back the justification test looks, in frames (0.5 s at 20 Hz).

Half a second is the timescale of a jerk-limited brake release, so it is the
shortest window over which "the brake is still coming off" and "the brake is
being newly applied for no reason" can be told apart.
"""

AVOIDANCE_HORIZON_S = 12.0
"""How far ahead the avoidability simulation looks.

From 33 m/s (the highest speed any scenario uses) full-authority braking
reaches standstill in 4.2 s; 12 s covers that plus a lead that is still rolling.
"""


# --------------------------------------------------------------------------- #
# Closed-form kinematics
# --------------------------------------------------------------------------- #


def travel_m(v0_mps: float, accel_mps2: float, t_s: float) -> float:
    """Distance covered in ``t_s`` by a body that cannot travel backwards.

    Constant acceleration, speed clamped at zero.  Exact, not integrated.
    """
    if t_s <= 0.0:
        return 0.0
    if accel_mps2 >= 0.0:
        return v0_mps * t_s + 0.5 * accel_mps2 * t_s * t_s
    t_stop = v0_mps / (-accel_mps2)
    if t_s <= t_stop:
        return v0_mps * t_s + 0.5 * accel_mps2 * t_s * t_s
    return v0_mps * v0_mps / (-2.0 * accel_mps2)


def min_gap_under_constant_decel(
    gap0_m: float,
    ego_v_mps: float,
    lead_v_mps: float,
    lead_a_mps2: float,
    ego_decel_mps2: float,
    horizon_s: float = AVOIDANCE_HORIZON_S,
) -> float:
    """Smallest gap reached if the ego decelerates at a constant rate.

    Both vehicles have piecewise-linear speed (constant acceleration, clamped at
    zero), so the gap is piecewise quadratic and its minimum can only occur at a
    phase boundary or where the relative speed passes through zero.  Those
    candidate times are enumerated exactly; no numerical integration is used and
    no timestep can hide a minimum.

    Args:
        gap0_m: Bumper-to-bumper gap now, metres.
        ego_v_mps: Ego speed now, m/s.
        lead_v_mps: Lead speed now, m/s.
        lead_a_mps2: Lead acceleration, held for the whole horizon (the causal
            assumption described in the module docstring).
        ego_decel_mps2: Constant ego deceleration, m/s^2, non-negative.
        horizon_s: How far ahead to look.

    Returns:
        The minimum gap over ``[0, horizon_s]``, metres.
    """
    a_ego = -abs(ego_decel_mps2)
    candidates = [0.0, horizon_s]
    if a_ego < 0.0:
        candidates.append(ego_v_mps / (-a_ego))
    if lead_a_mps2 < 0.0:
        candidates.append(lead_v_mps / (-lead_a_mps2))
    # Relative speed zero while both are still moving.
    rel_a = lead_a_mps2 - a_ego
    if abs(rel_a) > 1e-12:
        t_cross = (ego_v_mps - lead_v_mps) / rel_a
        candidates.append(t_cross)

    best = float("inf")
    for t in candidates:
        if t < 0.0 or t > horizon_s:
            continue
        gap = gap0_m + travel_m(lead_v_mps, lead_a_mps2, t) - travel_m(ego_v_mps, a_ego, t)
        best = min(best, gap)
    return best


def required_decel_mps2(
    gap0_m: float,
    ego_v_mps: float,
    lead_v_mps: float,
    lead_a_mps2: float,
    max_decel_mps2: float = DEFAULT_PLANT.max_brake_decel_mps2,
    clearance_m: float = REQUIRED_CLEARANCE_M,
) -> float:
    """Smallest constant deceleration that keeps ``clearance_m`` of room.

    Returns 0.0 when no braking is needed, and ``inf`` when even
    ``max_decel_mps2`` cannot preserve the clearance.

    The clearance target is capped at the gap that exists right now: once the
    ego is already closer than :data:`REQUIRED_CLEARANCE_M` (stopped in traffic,
    say) no deceleration can restore that clearance and demanding it would make
    the oracle report a permanent unavoidable emergency.

    This is an *idealised* number: it assumes the deceleration appears
    instantly.  Actuator lag is deliberately excluded here and modelled instead
    in :func:`last_avoidance_frame`, so that the justification test is a
    statement about kinematics and the lateness test is a statement about the
    vehicle.
    """
    if not math.isfinite(gap0_m):
        return 0.0
    closing = ego_v_mps - lead_v_mps
    if closing <= 1e-6 and lead_a_mps2 >= -1e-9:
        # Not closing and the lead is not slowing: nothing to brake for.
        return 0.0
    target = min(clearance_m, max(0.0, gap0_m - 0.05))
    if min_gap_under_constant_decel(gap0_m, ego_v_mps, lead_v_mps, lead_a_mps2, 0.0) >= target:
        return 0.0
    if (
        min_gap_under_constant_decel(
            gap0_m, ego_v_mps, lead_v_mps, lead_a_mps2, max_decel_mps2
        )
        < target
    ):
        return float("inf")
    lo, hi = 0.0, float(max_decel_mps2)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if min_gap_under_constant_decel(gap0_m, ego_v_mps, lead_v_mps, lead_a_mps2, mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


# --------------------------------------------------------------------------- #
# Avoidability, with the real actuator
# --------------------------------------------------------------------------- #


def full_braking_min_gap(
    state: WorldState,
    lead: Optional[LeadSpec],
    road: Optional[RoadSpec] = None,
    config: PlantConfig = DEFAULT_PLANT,
    horizon_s: float = AVOIDANCE_HORIZON_S,
) -> float:
    """Minimum gap if the ego applies full brake from ``state`` onward.

    Runs the SAME plant the system was driven by, including the brake actuator
    lag, and lets the lead continue its true script.  This is the physical
    answer to "could it still have stopped in time?", so it must not use an
    idealised instant-deceleration model.
    """
    if lead is None or not state.lead_present:
        return float("inf")
    sim = Plant.from_state(state, lead=lead, road=road, config=config)
    worst = state.gap_m
    steps = int(round(horizon_s / config.dt_s))
    for _ in range(steps):
        s = sim.step(0.0, 1.0, 0.0)
        if not s.lead_present:
            break
        worst = min(worst, s.gap_m)
        if s.ego_v_mps <= 1e-6 and s.lead_v_mps >= s.ego_v_mps:
            break
    return worst


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


@dataclass
class OracleVerdict:
    """What the kinematics say about a run, independent of the system."""

    frames: int = 0

    collided: bool = False
    collision_frame: Optional[int] = None
    min_gap_m: float = float("inf")

    required_decel: List[float] = field(default_factory=list)
    """Per-frame true required deceleration under the causal lead assumption."""

    emergency: bool = False
    """True when the true requirement reached :data:`COMFORT_DECEL_MPS2`."""

    first_emergency_frame: Optional[int] = None
    """First frame at which braking above comfort was genuinely required."""

    first_hazard_frame: Optional[int] = None
    """First frame at which ANY non-negligible braking was required."""

    last_avoidance_frame: Optional[int] = None
    """Last frame from which full-authority braking still avoided contact.

    ``None`` when there was never anything to avoid.  Intervening after this
    frame cannot prevent a collision, so an intervention that first appears at
    frame ``f > last_avoidance_frame`` is "late by ``f - last_avoidance_frame``
    frames" and the collision is the harness's fault to report, not the
    scenario's.
    """

    avoidable: Optional[bool] = None
    """Whether contact was avoidable at the first frame the lead was visible."""

    hazard_clear_frame: Optional[int] = None
    """First frame from which the requirement stays negligible for the rest of
    the run.  Recovery is measured from here."""

    quiet_frames: List[int] = field(default_factory=list)
    """Frames on which the true requirement was below
    :data:`NEGLIGIBLE_DECEL_MPS2`: no intervention is warranted on these."""

    def justified_decel_mps2(self, frame: int) -> float:
        """The largest deceleration that is defensible at ``frame``.

        The response must be COMMENSURATE with the requirement, over the whole
        range of the brake and not only at AEB grade.  The ceiling is the true
        requirement scaled by :data:`JUSTIFICATION_MARGIN_FACTOR` and offset by
        :data:`JUSTIFICATION_TOLERANCE_MPS2`, with a floor of
        :data:`HEADWAY_DECEL_ALLOWANCE_MPS2` -- the authority a following law
        may use when nothing at all is required.  ``inf`` when the situation is
        already unavoidable, at which point any amount of braking is
        defensible.

        The floor used to be :data:`EMERGENCY_DECEL_MPS2`, which left the entire
        sub-emergency band unpoliced: a system could brake at 3.4 m/s^2 for a
        lead at constant range forever and nothing in this specification would
        notice.  Dropping the floor to the headway allowance is what makes
        proportionality mean something -- at a true requirement of 1.0 m/s^2,
        the boundary of a hazard at all, the ceiling is 2.0 m/s^2, so three
        times the required brake is a finding rather than a rounding error.

        The requirement is taken as the WORST over the preceding
        :data:`JUSTIFICATION_WINDOW_FRAMES` frames, not just this one.  A brake
        that is doing its job makes its own justification disappear -- the gap
        stops shrinking -- and a jerk-limited brake cannot be released
        instantly, so judging each frame against only that frame's requirement
        would score every successful intervention as unjustified on the way out
        of it.
        """
        if not self.required_decel:
            return COMFORT_DECEL_MPS2
        lo = max(0, min(frame, len(self.required_decel) - 1) - JUSTIFICATION_WINDOW_FRAMES)
        hi = min(frame, len(self.required_decel) - 1)
        need = max(self.required_decel[lo : hi + 1])
        if math.isinf(need):
            return float("inf")
        return max(
            HEADWAY_DECEL_ALLOWANCE_MPS2,
            need * JUSTIFICATION_MARGIN_FACTOR + JUSTIFICATION_TOLERANCE_MPS2,
        )

    def emergency_warranted_at(self, frame: int) -> bool:
        """Whether a genuine emergency was live anywhere in the trailing window.

        "Emergency" here means the true requirement reached
        :data:`COMFORT_DECEL_MPS2`, the level above which braking stops being
        headway keeping and becomes collision avoidance.  The same trailing
        window as :meth:`justified_decel_mps2` is used, and for the same
        reason: an intervention that is working drives its own requirement to
        zero, and the frames in which it is still ramping the brake must be
        judged against the emergency it is answering, not against the calm it
        has just created.

        This selects which jerk ceiling applies -- see
        :func:`jerk_limit_mps3` -- because a rate of change of demand that
        buys stopping distance in an emergency buys nothing but discomfort
        outside one.
        """
        if not self.required_decel:
            return False
        hi = min(max(frame, 0), len(self.required_decel) - 1)
        lo = max(0, hi - JUSTIFICATION_WINDOW_FRAMES)
        return any(d >= COMFORT_DECEL_MPS2 for d in self.required_decel[lo : hi + 1])


    def warranted_at(self, frame: int) -> bool:
        """Whether braking was warranted at all on ``frame``."""
        if not self.required_decel:
            return False
        i = min(max(frame, 0), len(self.required_decel) - 1)
        return self.required_decel[i] >= NEGLIGIBLE_DECEL_MPS2

    def is_quiet(self, frame: int) -> bool:
        """True when nothing in the preceding half second warranted braking.

        Uses the same window as :meth:`justified_decel_mps2` and for the same
        reason: the last frames of a correct intervention have a low
        requirement precisely because the intervention worked.
        """
        if not self.required_decel:
            return True
        hi = min(frame, len(self.required_decel) - 1)
        lo = max(0, hi - JUSTIFICATION_WINDOW_FRAMES)
        return all(d < NEGLIGIBLE_DECEL_MPS2 for d in self.required_decel[lo : hi + 1])


def jerk_limit_mps3(emergency: bool) -> float:
    """The ceiling on the rate of change of demanded deceleration, m/s^3.

    Args:
        emergency: True when a genuine emergency was live in the trailing
            window, or when the system is still completing a stop that was
            warranted when it began.

    Returns:
        :data:`EMERGENCY_JERK_MPS3` or :data:`COMFORT_JERK_MPS3`.
    """
    return EMERGENCY_JERK_MPS3 if emergency else COMFORT_JERK_MPS3


def judge(
    history: Sequence[WorldState],
    lead: Optional[LeadSpec],
    road: Optional[RoadSpec] = None,
    config: PlantConfig = DEFAULT_PLANT,
) -> OracleVerdict:
    """Analyse a true state history and return the kinematic verdict.

    Args:
        history: The true states, frame 0 first.  This is the trajectory that
            ACTUALLY happened, so the verdict is about the run that occurred,
            not about a hypothetical one.
        lead: The lead script, needed for the counterfactual braking runs.
        road: The road, for the same reason.
        config: The plant configuration the run used.

    Returns:
        An :class:`OracleVerdict`.
    """
    v = OracleVerdict(frames=len(history))

    for s in history:
        gap = s.gap_m
        if math.isfinite(gap):
            v.min_gap_m = min(v.min_gap_m, gap)
            if gap <= CONTACT_GAP_M and v.collision_frame is None:
                v.collided = True
                v.collision_frame = s.frame

        v.required_decel.append(
            required_decel_mps2(
                gap0_m=gap,
                ego_v_mps=s.ego_v_mps,
                lead_v_mps=s.lead_v_mps,
                lead_a_mps2=s.lead_a_mps2,
                max_decel_mps2=config.max_brake_decel_mps2,
            )
            if s.lead_present
            else 0.0
        )

    for i, need in enumerate(v.required_decel):
        if need >= COMFORT_DECEL_MPS2 and v.first_emergency_frame is None:
            v.first_emergency_frame = i
        if need >= NEGLIGIBLE_DECEL_MPS2 and v.first_hazard_frame is None:
            v.first_hazard_frame = i
        if need < NEGLIGIBLE_DECEL_MPS2:
            v.quiet_frames.append(i)
    v.emergency = v.first_emergency_frame is not None

    # The hazard is over from the first frame whose requirement, and every
    # requirement after it, is negligible.
    clear: Optional[int] = None
    for i in range(len(v.required_decel) - 1, -1, -1):
        if v.required_decel[i] < NEGLIGIBLE_DECEL_MPS2:
            clear = i
        else:
            break
    if v.first_hazard_frame is not None and clear is not None and clear > v.first_hazard_frame:
        v.hazard_clear_frame = clear

    if v.first_hazard_frame is not None and lead is not None:
        last_ok: Optional[int] = None
        for s in history:
            if not s.lead_present:
                continue
            if full_braking_min_gap(s, lead, road, config) > CONTACT_GAP_M:
                last_ok = s.frame
            elif last_ok is not None:
                break
        v.last_avoidance_frame = last_ok
        first_visible = next((s for s in history if s.lead_present), None)
        if first_visible is not None:
            v.avoidable = full_braking_min_gap(first_visible, lead, road, config) > CONTACT_GAP_M

    return v
