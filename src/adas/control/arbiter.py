"""Authoritative safety arbitration for the ADAS decision layer.

``SafetyArbiter.arbitrate(plan, command, context)`` returns an
:class:`adas.core.models.ArbitrationResult`.  **The ``command`` field of that
result is what the actuators get.**

Evidence-gated authority
------------------------
The organising principle of this module, and of the longitudinal path it sits
at the end of, is that **the authority the system may exercise is a function of
the evidence supporting it.**  Concretely, three tiers, and the boundary between
them is a measurement rather than a threshold on a fabricated number:

============================  =========================================  ==================
evidence                      what it supports                           ceiling
============================  =========================================  ==================
no measured closing rate      nothing; the primary path's own following  no braking of the
                              law carries the frame                      arbiter's own
measured closure, but the     nothing yet; the arbiter reports the        no braking of the
4-sigma lower bound is not    situation and floors the state at LIMITED   arbiter's own
clear of zero
measured closure, lower       the deceleration the geometry requires,     full authority,
bound clear of zero           sized on the UNBIASED estimate              8.0 m/s^2
============================  =========================================  ==================

and one tier in the other direction, which is what makes this an authority
rather than a wire:

============================  =========================================  ==================
healthy perception, an EMPTY  nothing; a brake handed in from upstream    the arbiter VETOES
in-path corridor, a moving    has no support in any measurement           down to
ego                                                                       ``no_hazard_decel_cap_mps2``
============================  =========================================  ==================

**No rate is ever fabricated.**  A previous revision seeded a new track's range
rate at ``-ego_speed`` -- "assume the object is stationary in the world" -- so
that the backstop would not be blind on the frame a hazard appeared, and then
let the rate-dependent emergency tests read it.  That seed is a statement about
ignorance, and treating it as a measurement braked a 20 m/s ego to a standstill
behind a lead holding a constant 32.5 m.  The seed is gone.  What replaced it is
not a different number but a different structure: until four distinct captures
exist there is no rate at all, the arbiter says so
(:attr:`LeadAssessment.rate_is_measured`), and it adds no braking of its own.
The cost of that wait is two decision frames at 20 Hz -- the same two frames the
physics already charges, because at 55 ms of sense latency on a 50 ms grid
decision frames 0 and 1 read the SAME image and no closing rate exists before
frame 2 whatever the code does.

Independence
------------
A monitor that shares its inputs and its beliefs with the planner is a second
copy of the same belief, not a check.  What follows describes what the code in
this module actually does:

* **Its own in-path geometry.**  :meth:`SafetyArbiter._in_path` selects
  candidates from the object's own measured LATERAL OFFSET FROM THE EGO and from
  raw box positions in the arbiter's own image corridor.  It does NOT read
  ``TrackedObject.in_ego_lane``: that flag is computed by the tracker from the
  same lane model the planner uses, so trusting it would make the "independent"
  backstop inherit the very perception error it exists to catch -- and a lane
  fit that has slipped a metre toward the next lane reports a car that is
  nowhere near the ego as being directly in front of it.  Whether a collision is
  geometrically possible is decided by where the object is relative to the EGO,
  which the detector measures from the box and which no lane model can move.
  The image corridor is at least ``min_in_path_half_width_frac`` of the frame
  width and is anchored on the IMAGE CENTRE; a lane model widens it (a second
  anchor) only when it is real and confident, and can never narrow or move it.
* **Its own range evidence.**  It keeps its own window of RAW range
  measurements per track (:class:`adas.control.evidence.RangeEvidence`) and
  derives its own closing rate, its own standard error on that rate, and its own
  split-half estimate of the lead's acceleration.  It never reads
  ``TrackedObject.velocity_mps``.  A range discontinuity beyond
  ``evidence.jump_m`` re-anchors the window rather than being differentiated
  into a 200 m/s closure.
* **Its own hazard maths.**  The required deceleration and an RSS-style minimum
  gap are computed here from ``(range, measured closure, ego speed, lead
  acceleration)``; none of the planner's numbers are trusted.
* **A second range channel when one exists.** ``TrackedObject.range_estimate``
  and ``SafetyContext.independent_ranges`` carry a range from a source other than
  the box-height pinhole heuristic.  See :meth:`SafetyArbiter._fuse_range`.
* **Its own kinematics.**  Achieved acceleration and jerk are differenced from a
  least-squares fit of the measured ego speed, not inferred from the plan over a
  fictitious horizon.

What is NOT independent: ``SafetyContext.tracks`` is the tracker's output, so a
detection the perception stack never produced is invisible here too.  The
arbiter is a second opinion on the DECISION, not a second sensor.  What is also
shared, deliberately, is the *arithmetic* of measuring -- both this module and
:mod:`adas.planning.longitudinal` build their estimates with
:mod:`adas.control.evidence`.  They share no state, no lead selection and no
parameters; a least-squares slope is a numerical primitive, and having two
hand-written copies of one would buy diversity in the least valuable place and
pay for it with two sets of bugs.

The arbiter may never make the output less safe -- with one stated exception
-----------------------------------------------------------------------------
:meth:`SafetyArbiter._synthesise_command` never raises the throttle, and never
lowers the brake below the one it was handed **unless** the veto tier above
applies: healthy perception, a valid moving ego, and no object anywhere in the
arbiter's own (deliberately generous) in-path corridor.  In that one case a
brake handed in from upstream is not supported by any measurement the arbiter
can make, and passing it through would make the arbiter a wire with logging.
The exception is narrow on purpose: a single detection miss does not open it
(the corridor has to have been empty for the whole ``miss_hold_frames`` window
if the arbiter was braking), a perception dropout does not open it, and it can
never fire while the arbiter itself is demanding deceleration.

Findings: four kinds, only one of which can disengage
-----------------------------------------------------
Every frame produces up to four lists, all of which appear in
``ArbitrationResult.violations``:

``faults``
    HEALTH faults: the automation cannot perceive, cannot be commanded, or cannot
    tell the time.  ONLY these accumulate toward the terminal ``DISENGAGE``
    latch, because only these mean the system cannot continue driving.
``advisory``
    Findings that are a CONSEQUENCE OF THE ARBITER'S OWN INTERVENTION.  They are
    reported and they change nothing else.  **The arbiter must not treat a
    consequence of its own intervention as evidence that the intervention is
    still needed** -- that is a latch built out of its own actuator, and it is
    what stopped the vehicle on an open road.
``mitigated``
    Conditions the arbiter DETECTED AND HANDLED this frame, or measured and can
    only report.  They force ``LIMITED`` and are logged, but they NEVER latch
    ``DISENGAGE``.  A clamp that worked is normal operation.
``hazards``
    Traffic situations the arbiter exists to handle.  They force ``LIMITED`` (or
    ``MIN_RISK_MANEUVER`` when the measured closure warrants an emergency stop)
    and never latch ``DISENGAGE``: a two-second approach to a stopped queue must
    not disengage the system at exactly the wrong moment.

States
------
``SafetyState`` from ``core/models.py``, in increasing severity:

``NOMINAL``
    No finding at all.  The output command is the input command, made no more
    energetic.
``LIMITED``
    The arbiter modified or overrode the command but the function continues.
``MIN_RISK_MANEUVER``
    A measured closure warrants an emergency stop, or the vehicle has been blind
    long enough that it must come to a controlled stop.  Throttle is locked out
    and the steering path is HELD.
``DISENGAGE``
    Only after ``disengage_after_frames`` consecutive HEALTH faults.  It hands a
    moving vehicle back to nobody, so nothing short of a genuine loss of the
    decision path reaches it.

Units: metres, m/s, m/s^2, m/s^3, radians, seconds, and dimensionless actuator
commands in [0, 1] / [-1, 1].  Stateful and NOT thread-safe; one instance per
pipeline, and call :meth:`SafetyArbiter.reset` on replay restart.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from adas.control.evidence import (
    ClosureEstimate,
    EvidenceBook,
    EvidenceLimits,
    JerkShaper,
    required_decel_mps2,
    sub_emergency_guard,
)
from adas.core.logger import Throttle, setup_logger
from adas.core.models import (
    ArbitrationResult,
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

logger = setup_logger(__name__)

NOMINAL_DT_S = 0.05

_SEVERITY = {
    SafetyState.NOMINAL: 0,
    SafetyState.LIMITED: 1,
    SafetyState.MIN_RISK_MANEUVER: 2,
    SafetyState.DISENGAGE: 3,
}


_NUMBER_RUN = re.compile(r"\d+(?:\.\d+)?")
"""Any run of digits in a violation string, with an optional decimal part."""


def _condition_key(violations: List[str]) -> Tuple[str, ...]:
    """The IDENTITY of a set of violations, with every number replaced by ``#``.

    Used only to decide whether a log line is a CHANGE or a repeat.  Violation
    strings carry their measurements -- ``perception_dropout_1184``,
    ``headway_14.7m_below_rss_25.1m`` -- so comparing them literally makes every
    frame of one unchanging condition look like a transition, and a log throttle
    keyed on that throttles nothing.  Measured: 1200 lines for 1200 frames of a
    single continuous perception dropout, which is the exact defect the throttle
    exists to prevent, reconstructed inside the fix for it.

    Normalising the numbers away keeps the useful discrimination and drops the
    useless one: a NEW KIND of violation always logs immediately, a counter
    ticking or a range drifting does not.  The numbers are still in every line
    that is emitted, so nothing is lost from the record.
    """
    return tuple(sorted(_NUMBER_RUN.sub("#", v) for v in violations))


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``; ``low`` wins if the interval is empty."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def _worse(a: SafetyState, b: SafetyState) -> SafetyState:
    """Return whichever of the two states is more severe."""
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


# --------------------------------------------------------------------------- #
# Configuration and inputs
# --------------------------------------------------------------------------- #


@dataclass
class ArbiterLimits:
    """Hard limits and thresholds. These must be STRICTER than the planner's."""

    # Longitudinal envelope
    max_speed_mps: float = 33.0
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 8.0
    comfort_decel_mps2: float = 3.0
    mrm_decel_mps2: float = 3.0
    """Deceleration of a minimum-risk stop, m/s^2.

    The comfort limit, and deliberately BELOW ``emergency_grade_mps2``: an MRM is
    a controlled stop because nothing has been DETECTED in front of the vehicle
    -- the reason to stop is that the vehicle cannot see, not that something is
    there -- and the traffic behind has no reason to expect an AEB.  A
    minimum-risk manoeuvre that brakes like an emergency turns every perception
    dropout into a rear-end risk.
    """
    max_jerk_mps3: float = 4.0
    """RETAINED AND UNUSED BY THE DECISION, m/s^3.

    It was the comfort ceiling on the ACHIEVED jerk -- the jerk differenced from
    the measured ego speed.  The redesign judges the arbiter's OWN demand
    instead, in :meth:`SafetyArbiter._check_command_jerk`, against
    ``comfort_jerk_mps3`` and ``emergency_jerk_mps3``, because a jerk
    differentiated twice from a filtered wheel speed is dominated by the
    estimator and is unactionable: the vehicle cannot be told to have been
    smoother a frame ago.  Kept so an existing ``SafetyConfig`` still loads."""
    max_jerk_emergency_mps3: float = 25.0
    """RETAINED AND UNUSED BY THE DECISION, m/s^3.  See ``max_jerk_mps3``.

    Its original argument, preserved because the number is still the right one
    should an achieved-jerk report ever come back:

    The specification's own emergency demand ceiling is 20 m/s^3; the vehicle
    answering a compliant 20 m/s^3 demand through a 0.15 s brake rise, measured
    by differentiating a filtered wheel speed, lands just under that with a few
    m/s^3 of estimator ripple on top.  Reporting the vehicle for obeying a legal
    command would be the arbiter indicting itself, so the reported ceiling
    carries the estimator's own margin.
    """
    accel_authority_mps2: float = 2.5
    """RETAINED AND UNUSED BY THE DECISION, m/s^2 delivered at throttle = 1.0.

    The arbiter never converts an acceleration into a throttle: it only ever
    REDUCES a throttle it was handed, so it needs the authority constant for the
    brake and not for the pedal it cannot press.  Kept so an existing
    ``SafetyConfig`` still loads, and range-checked below."""
    brake_authority_mps2: float = 8.0
    """m/s^2 delivered at brake = 1.0. Used to turn a required deceleration into a
    pedal fraction; must match the vehicle the commands are sent to."""

    # Lateral envelope
    max_steering_angle_rad: float = 0.52
    max_steering_rate_rad_s: float = 0.5
    max_lateral_accel_mps2: float = 4.5
    max_lateral_offset_m: float = 1.5
    """Lane-departure limit, metres from the lane centre. Enforceable ONLY when
    ``SafetyContext.lateral_offset_m`` is supplied by a calibrated lane geometry;
    without it the arbiter reports ``lane_offset_unavailable`` rather than
    pretending the check ran."""
    wheelbase_m: float = 2.8
    max_road_wheel_rad: float = 0.436
    """Road-wheel angle corresponding to ``steering = 1.0`` (25 deg by default).
    Must match ``ControllerConfig.max_steering_angle_deg`` or the normalised
    command cannot be converted to a physical angle."""

    # Headway / collision avoidance
    absolute_min_gap_m: float = 2.0
    standstill_gap_m: float = 4.0
    """RETAINED AND UNUSED BY THE DECISION, metres.

    The clearance an avoidance stop aims at is ``target_clearance_m`` (2.25 m)
    and the clearance the WARRANT test is written against is
    ``warrant_clearance_m`` (2.0 m); both are read, both are named for what they
    do, and neither is this.  It is still cross-checked against
    ``absolute_min_gap_m`` below so a deployment that sets an incoherent pair is
    told."""
    reaction_time_s: float = 0.6
    ego_brake_capability_mps2: float = 6.0
    lead_brake_capability_mps2: float = 8.0
    ttc_brake_s: float = 0.9
    ttc_warn_s: float = 1.6
    ttc_min_closing_mps: float = 0.5
    """Closure below which no time to contact is computed at all, m/s.

    A gap divided by a rate of a few centimetres a second is a large number that
    means nothing, and a gap divided by half a metre a second is a small number
    that means a queue.  Neither is a time to collision.  Half a metre per second
    is also the speed below which this module already declines to differentiate
    the ego bus, so the two floors agree.
    """
    aeb_required_decel_mps2: float = 5.0
    """Required deceleration above which the arbiter calls a MEASURED situation an
    emergency and latches a minimum-risk manoeuvre.  It does not gate the pedal:
    the braking a measured closure demands goes out on the frame it is seen."""
    aeb_decel_margin: float = 1.05
    aeb_min_decel_mps2: float = 0.0
    """Floor on the emergency demand, m/s^2.

    ZERO, and that is the fix for a measured defect rather than a disabled
    check.  A floor forces the demand up to a fixed number the moment any
    emergency predicate trips, so the response stops being proportional to the
    situation at exactly the point where proportionality matters most -- and a
    4.0 m/s^2 floor applied 40 m from a stopped car where 2.7 m/s^2 was required
    is over-braking, which transfers the collision to the vehicle behind.  The
    geometric requirement already rises to full authority on its own when the
    geometry asks for it.
    """
    aeb_headway_frac: float = 0.5
    """A gap below this fraction of the RSS minimum is reported as a hazard in its
    own right, because even ordinary braking by the lead would then cause a
    collision.  It floors the STATE at LIMITED; it does not authorise braking,
    because a short gap is not a closure and closing the gap is the following
    law's job."""

    # --- the evidence gate ----------------------------------------------------
    evidence: EvidenceLimits = field(default_factory=EvidenceLimits)
    """How much measurement the avoidance law demands before it engages at all.

    See :class:`adas.control.evidence.EvidenceLimits`.  The arbiter owns its own
    instance: the planner's setting can never reach it.
    """
    target_clearance_m: float = 2.25
    """Room a completed avoidance stop aims to leave, metres.

    The 2.0 m a driver leaves at standstill plus a quarter of a metre of aim-off,
    which is the range the arbiter cannot see -- it is acting on a measurement
    one frame old, so a law that aims at exactly the required clearance stops a
    fraction inside it every time.
    """
    demand_margin_frac: float = 0.05
    demand_margin_mps2: float = 0.05
    """Prudence added to the computed requirement.  Deliberately small: the
    requirement is recomputed every frame from a fresh measurement, so a standing
    margin buys nothing the next frame does not buy anyway, and an over-sized one
    is measured as over-braking."""
    warrant_clearance_m: float = 2.0
    """Clearance the WARRANT test is written against, metres.

    The specification's own ``REQUIRED_CLEARANCE_M``: the room a correct
    intervention preserves, and therefore the room against which "has an
    emergency arisen?" is asked.  It is deliberately smaller than
    ``target_clearance_m``, which is what the response AIMS at -- asking the
    warrant question against the aim point would declare an emergency slightly
    before the physics does.
    """
    band_guard_margin_mps2: float = 0.05
    """How far below the comfort limit an UNWARRANTED demand is held, m/s^2.

    See :func:`adas.control.evidence.sub_emergency_guard`.  The band is closed at
    the bottom -- a command of exactly 3.0 m/s^2 is inside it -- so the guard has
    to leave a gap, and 0.05 m/s^2 is the quantisation of a rate-limited brake
    command.
    """
    emergency_grade_mps2: float = 3.5
    """At or above this a command is an emergency intervention.  Half a metre
    above the comfort limit, so a comfort-limited ramp cannot be mistaken for an
    AEB by rounding."""
    comfort_jerk_mps3: float = 2.5
    """Rate the arbiter's own demand may be built at inside the comfort band."""
    emergency_jerk_mps3: float = 20.0
    """Rate the arbiter's own demand may be built at once it is heading past
    emergency grade: full authority in the 0.4 s a human panic brake takes.
    Faster buys no stopping distance -- the brake's own 0.15 s rise filters it
    out -- and costs the occupant a head-toss they cannot brace for."""
    release_jerk_mps3: float = 25.0
    """Rate the arbiter's own demand falls at.  Only the RISE is a comfort
    hazard, and every other requirement demands that an unwarranted deceleration
    be removed PROMPTLY: a brake that is still coming off half a second after
    the requirement collapsed is measured by this specification as
    over-braking, and over-braking transfers the collision to the vehicle
    behind.  25 m/s^3 clears full authority in 0.32 s, which is faster than
    the brake's own hydraulic decay, so the release the road sees is set by
    the plumbing rather than by this number."""
    blind_hold_frames: int = 8
    """Frames of perception loss tolerated before a minimum-risk stop begins.
    0.4 s, which at 20 m/s is 8 m travelled without a picture."""
    miss_hold_frames: int = 24
    """Frames the arbiter's own avoidance demand survives a DETECTION miss with
    healthy perception, 1.2 s.  A detector that produced nothing has not produced
    evidence of an empty road."""
    no_hazard_decel_cap_mps2: float = 0.0
    """Deceleration the arbiter will pass through on a demonstrably EMPTY road.

    The veto tier.  With healthy perception, a valid moving ego and no object
    anywhere in the arbiter's own corridor, a brake arriving from upstream is
    supported by no measurement at all, and the arbiter is the last component
    before the actuators.  Zero, because there is nothing to brake for: an
    unwarranted full-authority stop on a motorway does not avoid a collision, it
    manufactures one behind.  Raise it only for a deployment that has some other
    reason to decelerate on an empty road.
    """
    ego_half_width_m: float = 0.9
    """Half the ego's own width, metres.  Half of a 1.8 m passenger car."""
    lateral_gate_margin_m: float = 0.30
    """Extra half width the arbiter's METRIC in-path gate carries over the
    geometric footprint overlap, metres.

    It exists so that the backstop's corridor is strictly WIDER than the primary
    path's: the arbiter must never consider a narrower slice of the road than the
    thing it is backing up, or a hazard could be visible to neither.
    """

    log_repeat_period_s: float = 30.0
    """Seconds between repeats of the SAME sustained non-nominal log line.

    A degraded state that is a steady condition -- a vehicle with no CAN bus, a
    camera that has stopped, a lead held at a short gap in a queue -- used to
    produce one WARNING per frame: 1200 lines in a 1200-frame run, measured, and
    72,000 an hour at 20 Hz.  That has been shipped twice.  It fills the disk and
    it buries the transition that actually mattered.

    What is NEVER throttled is a CHANGE: every entry into a state and every exit
    from one is logged at full severity on the frame it happens, and so is every
    change in the set of violations, because those are the events an operator
    needs and there is one per transition rather than one per frame.  Only the
    unchanged repeat is rate limited, and the line that gets through carries the
    number of frames suppressed since the last one, so a throttled condition can
    never be mistaken for a quiet one.

    Set to 0.0 to log every frame; the state transitions are logged either way.
    """

    max_coast_frames: int = 5
    """How many frames the arbiter will keep assessing a track the tracker is
    COASTING (``time_since_update > 0``).  A coasted range is never folded into
    the measured-rate window: an extrapolation is not a measurement and must not
    be able to establish the closure that authorises an emergency stop."""

    # Plan feasibility
    plan_horizon_s: float = 1.0
    """RETAINED AND UNUSED BY THE DECISION, seconds.

    It was the horizon over which a plan's requested ACCELERATION was judged.
    The plan check now compares the requested target speed against the measured
    ego speed and the acceleration limits directly, with no horizon, because a
    horizon multiplied an already-bounded rate into a second bound that could
    disagree with the first.  Kept so an existing ``SafetyConfig`` still loads."""

    # Degradation policy
    limited_after_dropouts: int = 1
    mrm_after_dropouts: int = 8
    """Consecutive perception dropouts before a minimum-risk manoeuvre begins.

    Aligned with ``blind_hold_frames``: a dropped frame is a dropped frame, and
    the vehicle holds what it was doing.  Half a second of nothing is a vehicle
    driving blind and it must stop.  The previous value of 3 declared a
    minimum-risk manoeuvre -- with its throttle lock, its steering hold and its
    ten-frame recovery -- for a 0.15 s blink, which on real footage latched a
    degraded state that outlived every cause it had.
    """
    disengage_after_frames: int = 40
    recovery_frames: int = 10
    limited_throttle_cap: float = 0.0
    """Maximum throttle allowed once degraded. 0.0 means: never accelerate while
    the arbiter is not in NOMINAL."""

    kinematics_min_speed_mps: float = 0.5
    """Below this speed the achieved-acceleration and jerk checks are disabled.
    Wheel-speed quantisation and the standstill transition dominate the difference
    quotient there."""
    kinematics_streak_frames: int = 3
    """Consecutive frames an achieved-kinematics exceedance must persist before it
    is reported.

    The check differentiates the vehicle bus, and a bus with noise on it produces
    single-frame excursions that say nothing about the control.  A real
    over-acceleration lasts; a draw from the noise does not.
    """

    # In-path corridor -- the arbiter's OWN geometry, never the tracker's flag
    min_in_path_half_width_frac: float = 0.30
    """Floor on the half width of the arbiter's IMAGE corridor, as a fraction of
    the frame width.  The corridor actually used is
    ``max(this, SafetyContext.ego_lane_half_width_frac)``, so the arbiter's image
    window is always at least as wide as the planner's."""
    lane_trust_confidence: float = 0.50
    """Minimum ``LaneModel.confidence`` (and the lane must not be ``is_mock``)
    before the lane centre may be used as a SECOND corridor anchor.  A lane centre
    is never allowed to move or narrow the corridor."""
    mrm_straighten_speed_mps: float = 1.0
    """Below this speed a minimum-risk manoeuvre stops holding the steering angle
    and slews toward straight.  Above it the MRM HOLDS the last actuated angle:
    zeroing the wheel mid-corner is a lateral hazard in its own right."""

    # Timing
    min_dt_s: float = 0.005
    """Frames faster than this have their dt clamped up, silently. Running well is
    not a fault."""
    max_dt_s: float = 0.5
    max_frame_gap_s: float = 0.5

    # Diverse range channel
    range_disagreement_frac: float = 0.30
    """Relative disagreement between the two range channels that is reported and
    forces LIMITED. It is a mitigated condition, not a health fault.

    Reporting only.  The fused range is the NEARER of the two channels whether
    they agree or not (see :meth:`SafetyArbiter._fuse_range`), so this threshold
    changes what is said about a frame and never what is done on it.
    """
    min_range_confidence: float = 0.35
    """A second-channel :class:`~adas.core.models.RangeEstimate` below this
    confidence is treated as UNAVAILABLE rather than merely down-weighted."""

    # Camera calibration honesty
    allow_uncalibrated_range: bool = False
    """When False (the default) a ``SafetyContext.camera_calibrated=False`` frame
    reports ``camera_uncalibrated`` and the state can never be NOMINAL: the metric
    range every hazard test is built on is then an assumption, not a measurement."""

    # Output shaping (does not by itself escalate the state)
    throttle_rate_per_s: float = 5.0
    brake_release_rate_per_s: float = 8.0
    """Rate at which a brake INHERITED FROM UPSTREAM may be released.  There is
    deliberately no corresponding apply-rate limit: the arbiter must never
    attenuate a brake it did not decide to veto."""

    # --- retained for configuration compatibility -----------------------------
    deferred_aeb_decel_mps2: float = 5.0
    """RETAINED AND UNUSED BY THE DECISION.  It named the ceiling on braking
    driven by the ``-ego_speed`` safe prior while a rate was awaited.  There is no
    such prior any more, so there is nothing for it to bound; it is kept so that
    an existing ``SafetyConfig`` still loads, and it is validated so that a
    deployment which sets it is told the truth by
    :meth:`SafetyArbiter.unused_limits` rather than believing it took effect."""
    aeb_rate_corroboration_frames: int = 2
    """RETAINED AND UNUSED BY THE DECISION.  It named the consecutive frames a
    measured emergency had to hold before it latched a minimum-risk manoeuvre.

    Removed because it was a LATCH on evidence and it was redundant.  The thing
    it protected against -- one noisy frame latching a manoeuvre with a throttle
    lock and a ten-frame recovery -- is already carried, and carried better, by
    the two statistical gates underneath it: a closure must clear
    ``evidence.closure_sigmas`` standard errors of zero before it authorises any
    avoidance braking, and a lead deceleration must clear
    ``evidence.accel_sigmas`` before it is credited at all.  A frame counter on
    top of those buys no extra rejection (the noise that survives four sigma is
    not one frame long) and costs one frame on every genuine emergency.
    Measured on the sweep's tightest family (rate -8, lead_decel 6): the first
    minimum-risk frame moved from 4 to 3 with no new phantom, early or band
    finding anywhere in the 120-cell gate or the 52-scenario corpus.

    It is kept so that an existing ``SafetyConfig`` still loads, and it is
    reported by :meth:`SafetyArbiter.unused_limits`."""
    range_corroboration_frames: int = 3
    """RETAINED AND UNUSED BY THE DECISION.  It named the consecutive disagreeing
    frames required before a NEARER second-channel range was adopted.

    Removed with the rest of the range-fusion state machine: the fused range is
    now ``min(pinhole, second_channel)`` on every frame, so a nearer channel is
    adopted immediately -- which is the safe direction and was always immediate
    when the two channels AGREED, so the delay only ever applied to the frames
    where the second channel was most worth listening to."""
    range_source_dwell_frames: int = 5
    """RETAINED AND UNUSED BY THE DECISION.  Consecutive frames the second channel
    had to be unusable before the arbiter stopped using it."""
    range_confidence_hysteresis: float = 0.10
    """RETAINED AND UNUSED BY THE DECISION.  Hysteresis on the second channel's
    confidence gate."""
    range_disagreement_hysteresis: float = 0.25
    """RETAINED AND UNUSED BY THE DECISION.  Hysteresis band around
    ``range_disagreement_frac``, which is itself now reporting-only."""
    aeb_min_rate_samples: int = 3
    """Alias of ``evidence.min_samples``, projected onto it in ``__post_init__``
    so that an existing ``SafetyConfig`` keeps controlling the same thing."""
    aeb_min_rate_span_s: float = 0.10
    """Alias of ``evidence.min_span_s``."""
    range_rate_window_s: float = 0.45
    """Length of the raw-measurement window, seconds.  Projected onto
    ``evidence.window_samples`` at the nominal frame period."""

    def __post_init__(self) -> None:
        if self.log_repeat_period_s < 0.0:
            raise ValueError("log_repeat_period_s must be >= 0")
        if self.max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        if self.brake_authority_mps2 <= 0 or self.accel_authority_mps2 <= 0:
            raise ValueError("actuator authorities must be positive")
        if self.max_road_wheel_rad <= 0:
            raise ValueError("max_road_wheel_rad must be positive")
        if self.standstill_gap_m < self.absolute_min_gap_m:
            raise ValueError("standstill_gap_m must be >= absolute_min_gap_m")
        if self.ttc_warn_s < self.ttc_brake_s:
            raise ValueError("ttc_warn_s must be >= ttc_brake_s")
        if self.mrm_after_dropouts < self.limited_after_dropouts:
            raise ValueError("mrm_after_dropouts must be >= limited_after_dropouts")
        if self.recovery_frames < 1 or self.disengage_after_frames < 1:
            raise ValueError("recovery_frames and disengage_after_frames must be >= 1")
        if not (0.0 < self.aeb_headway_frac <= 1.0):
            raise ValueError("aeb_headway_frac must be in (0, 1]")
        if self.min_in_path_half_width_frac < 0.0:
            raise ValueError("min_in_path_half_width_frac must be >= 0")
        # The retained-compatibility keys are still range-checked, so that a
        # deployment which sets one to nonsense is told, even though nothing
        # reads it.  What it is NOT told here is that the key is inert; that is
        # :meth:`SafetyArbiter.unused_limits`, printed at start-up.
        if self.range_corroboration_frames < 1:
            raise ValueError("range_corroboration_frames must be >= 1")
        if self.aeb_rate_corroboration_frames < 1:
            raise ValueError("aeb_rate_corroboration_frames must be >= 1")
        if self.aeb_min_rate_samples < 3:
            raise ValueError("aeb_min_rate_samples must be >= 3")
        if self.aeb_min_rate_span_s <= 0.0:
            raise ValueError("aeb_min_rate_span_s must be > 0")
        if self.range_rate_window_s < self.aeb_min_rate_span_s:
            raise ValueError("range_rate_window_s must be >= aeb_min_rate_span_s")
        if self.range_source_dwell_frames < 1:
            raise ValueError("range_source_dwell_frames must be >= 1")
        if not (0.0 <= self.range_disagreement_hysteresis < 1.0):
            raise ValueError("range_disagreement_hysteresis must be in [0, 1)")
        if self.range_confidence_hysteresis < 0.0:
            raise ValueError("range_confidence_hysteresis must be >= 0")
        if self.max_coast_frames < 0:
            raise ValueError("max_coast_frames must be >= 0")
        if not (
            self.comfort_decel_mps2
            <= self.deferred_aeb_decel_mps2
            <= self.max_deceleration_mps2
        ):
            raise ValueError(
                "deferred_aeb_decel_mps2 must be between comfort_decel_mps2 and "
                "max_deceleration_mps2"
            )
        if self.min_dt_s <= 0.0 or self.max_dt_s <= self.min_dt_s:
            raise ValueError("max_dt_s must be > min_dt_s > 0")
        if not (
            self.comfort_decel_mps2
            <= self.emergency_grade_mps2
            <= self.max_deceleration_mps2
        ):
            raise ValueError(
                "emergency_grade_mps2 must lie between comfort_decel_mps2 and "
                "max_deceleration_mps2"
            )
        if self.mrm_decel_mps2 <= 0.0 or self.mrm_decel_mps2 > self.max_deceleration_mps2:
            raise ValueError("mrm_decel_mps2 must be in (0, max_deceleration_mps2]")
        if self.comfort_jerk_mps3 <= 0.0 or self.emergency_jerk_mps3 < self.comfort_jerk_mps3:
            raise ValueError("require 0 < comfort_jerk_mps3 <= emergency_jerk_mps3")
        if self.release_jerk_mps3 <= 0.0:
            raise ValueError("release_jerk_mps3 must be positive")
        if self.target_clearance_m < 0.0 or self.warrant_clearance_m < 0.0:
            raise ValueError("clearance targets must be >= 0")
        if self.warrant_clearance_m > self.target_clearance_m:
            raise ValueError(
                "warrant_clearance_m must not exceed target_clearance_m, or the "
                "warrant test would fire before the response it authorises aims at "
                "anything"
            )
        if not 0.0 < self.band_guard_margin_mps2 < self.comfort_decel_mps2:
            raise ValueError("band_guard_margin_mps2 must be in (0, comfort_decel_mps2)")
        if self.no_hazard_decel_cap_mps2 < 0.0:
            raise ValueError("no_hazard_decel_cap_mps2 must be >= 0")
        if self.aeb_min_decel_mps2 < 0.0:
            raise ValueError("aeb_min_decel_mps2 must be >= 0")
        if self.ego_half_width_m <= 0.0 or self.lateral_gate_margin_m < 0.0:
            raise ValueError("ego_half_width_m must be > 0 and the gate margin >= 0")
        if self.blind_hold_frames < 0 or self.miss_hold_frames < 0:
            raise ValueError("hold windows must be non-negative")
        if self.kinematics_streak_frames < 1:
            raise ValueError("kinematics_streak_frames must be >= 1")

        # Project the configuration-visible aliases onto the evidence limits, so
        # that a deployment tuning ``SafetyConfig`` is tuning the thing it names
        # rather than a field the decision no longer reads.
        window = max(
            2 * int(self.aeb_min_rate_samples),
            int(round(self.range_rate_window_s / NOMINAL_DT_S)) + 1,
        )
        self.evidence = EvidenceLimits(
            window_samples=window,
            min_samples=int(self.aeb_min_rate_samples),
            min_span_s=float(self.aeb_min_rate_span_s),
            closure_sigmas=self.evidence.closure_sigmas,
            accel_sigmas=self.evidence.accel_sigmas,
            jump_m=self.evidence.jump_m,
            sigma_ewma_alpha=self.evidence.sigma_ewma_alpha,
            max_lead_decel_mps2=self.max_deceleration_mps2,
        )

    def unused_limits(self) -> Tuple[str, ...]:
        """Fields kept for configuration compatibility that the decision ignores.

        A limit that a deployment can set and that nothing reads is worse than an
        absent one: it reads as control and delivers none.  This method is what a
        start-up check should print.

        It is checked in BOTH directions by
        ``tests/test_arbiter.py::test_unused_limits_is_truthful_in_both_directions``,
        which inspects this module's source rather than trusting a hand-maintained
        list: a name here that the decision does read is a lie, and a field the
        decision does not read that is missing from here is a config key that
        silently does nothing.  Five of the eleven below were found by that test
        rather than by anyone remembering, and two of them still carried a
        docstring describing behaviour that had been removed.

        Fields that reach the decision by PROJECTION rather than by being read --
        ``aeb_min_rate_samples``, ``aeb_min_rate_span_s`` and
        ``range_rate_window_s``, which ``__post_init__`` folds into
        ``evidence`` -- are live and are deliberately absent from this list.
        """
        return (
            "accel_authority_mps2",
            "aeb_rate_corroboration_frames",
            "deferred_aeb_decel_mps2",
            "max_jerk_emergency_mps3",
            "max_jerk_mps3",
            "plan_horizon_s",
            "range_confidence_hysteresis",
            "range_corroboration_frames",
            "range_disagreement_hysteresis",
            "range_source_dwell_frames",
            "standstill_gap_m",
        )


@dataclass
class SafetyContext:
    """Everything the arbiter needs that is not the plan or the command.

    This is the ``state`` argument of ``arbitrate(plan, cmd, state)``.

    Attributes:
        ego: Measured ego state. ``ego.valid`` must be True for ``speed_mps`` to be
            believed; an invalid or missing ego state forces a minimum-risk
            manoeuvre, because none of the hazard maths is defined without it.
        tracks: The RAW track list for this frame. The arbiter does its own lead
            selection over it. An empty list means "perception ran and saw
            nothing"; a perception failure must be signalled through
            ``perception``, never by passing an empty list.
        perception: Perception health for this frame.
        dt_s: Nominal timestep, seconds. Used only when ``timestamp_s`` is absent.
        timestamp_s: Monotonic capture time of this frame, seconds. When supplied
            on consecutive frames the arbiter uses the MEASURED interval instead of
            the nominal one, and flags a stale input if the gap is implausible.
        frame_width_px / frame_height_px: Image size, for the in-path gate.
        lane: Lane model. It may only WIDEN the arbiter's in-path corridor, by
            adding a second anchor, and only when it is real
            (``not is_mock``) and its confidence reaches
            ``ArbiterLimits.lane_trust_confidence``. It can never move or narrow
            the corridor: a lane-detection error must not be able to hide a lead.
        ego_lane_half_width_frac: Half width of the ego-lane band as a fraction of
            the frame width, usually the planner's value. The arbiter widens it to
            at least ``ArbiterLimits.min_in_path_half_width_frac``.
        independent_ranges: Range estimates from a channel other than the box
            height pinhole, keyed by track id.
        lateral_offset_m: Metric EGO offset from the lane centre, or None.  Used
            only for the lane-departure report; it is never used to decide
            whether an object is in the path, because that would put the lane
            estimator back inside the hazard gate.
        camera_calibrated: Whether the camera whose geometry produced the metric
            ranges is calibrated.
    """

    ego: EgoState | None = None
    tracks: List[TrackedObject] = field(default_factory=list)
    perception: PerceptionStatus | None = None
    dt_s: float = NOMINAL_DT_S
    timestamp_s: float | None = None
    measurement_t_s: float | None = None
    """CAPTURE time of the image the tracks in this frame were measured from,
    seconds; None when the caller does not know it.

    Distinct from ``timestamp_s``, and the distinction is load-bearing for every
    rate this arbiter fits.  A range history is a set of (time, range) pairs and
    a slope is only as good as its abscissae: the times must be when the ranges
    were TRUE, not when the decision loop got round to them.  With a constant
    sense latency the two differ by a constant and a slope does not care -- which
    is why this went unnoticed at the measured 55 ms -- but latency is not
    constant, and once it stretches past a frame period the SAME capture is
    republished on consecutive decision frames.  Deduplicating those (which this
    arbiter does, on ``TrackedObject.hits``) leaves the surviving samples stamped
    with the decision time of first sight, so three captures 50 ms apart are
    fitted as if they were 0, 150 and 200 ms apart.

    Measured, on ``degraded_latency_stationary_20mps_at_36m`` (a 36 m stationary
    obstacle at 20 m/s through 80 ms of sense latency): fitting on the decision
    clock reported a closing rate of 9.23 m/s for a true 20 m/s, the residuals of
    the misplaced fit inflated the standard error enough to close the
    four-sigma gate for two further frames, the first emergency-grade command
    arrived at frame 9 instead of frame 7, and the run finished 1.17 m from the
    obstacle against the 2.00 m required and the 3.70 m reachable.

    ``adas.runtime.pipeline`` passes ``frame.timestamp_s`` for both, which is the
    capture time, so production behaviour is unchanged by this field existing.
    """
    frame_width_px: int = 0
    frame_height_px: int = 0
    lane: LaneModel | None = None
    ego_lane_half_width_frac: float = 0.30
    independent_ranges: Dict[int, RangeEstimate] | None = None
    lateral_offset_m: float | None = None
    """Ego offset from the lane centre in METRES, positive when the ego is right
    of centre. Requires a calibrated camera; pass None when only a pixel-space lane
    centre is available and the arbiter will report the check as unavailable."""
    camera_calibrated: bool | None = None
    """True/False when the caller knows, None when it does not report."""
    camera_focal_px: float | None = None
    """Horizontal focal length of the camera the boxes came from, pixels.

    Used only to turn a box WIDTH into a metric width for the in-path gate.  When
    it is absent the arbiter falls back to the focal length implied by a 60 deg
    horizontal field of view and floors the result at half a passenger car, so a
    missing calibration can only ever WIDEN the corridor -- see
    :meth:`SafetyArbiter._object_half_width_m`.
    """


@dataclass
class LeadAssessment:
    """The arbiter's own view of the most dangerous in-path object.

    Every field here is either a MEASUREMENT or a statement about how well
    measured something is.  Nothing in it is a prior.
    """

    track_id: int
    distance_m: float
    range_rate_mps: float
    """Closing rate in the ``v_lead - v_ego`` convention: NEGATIVE while closing.
    0.0 while :attr:`rate_is_measured` is False, which means "not measured", not
    "not closing"."""
    ttc_s: float
    required_decel_mps2: float
    rss_min_gap_m: float
    source: RangeSource
    rss_matched_gap_m: float = 0.0
    """The RSS minimum gap computed with the OPTIMISTIC assumption that the lead is
    travelling at exactly the ego speed.  It depends on the measured range and the
    measured ego speed only -- no closing rate at all."""
    disagreement: bool = False
    reinitialised: bool = False
    coasting: bool = False
    coast_frames: int = 0
    rate_is_measured: bool = False
    """True once the closing rate is derived from the arbiter's own window of RAW
    range measurements.  While False the arbiter adds NO braking of its own: a
    quantity that has not been measured cannot authorise an action."""
    measured_rate_mps: float | None = None
    """The raw least-squares slope (``v_lead - v_ego``), or None."""
    rate_updates: int = 0
    """Distinct captures in the arbiter's own window for this track."""
    closing_lcb_mps: float = 0.0
    """Lower confidence bound on the closure, POSITIVE WHEN CLOSING.  This is what
    the braking law is GATED on; :attr:`range_rate_mps` is what it is SIZED on."""
    closing_stderr_mps: float = 0.0
    lead_accel_mps2: float = 0.0
    """Confident lead acceleration, <= 0.  0.0 means "no credible deceleration",
    never "the lead is not braking"."""
    lateral_offset_m: float | None = None
    """The object's measured lateral offset FROM THE EGO, metres, or None."""


# --------------------------------------------------------------------------- #
# The arbiter
# --------------------------------------------------------------------------- #


class SafetyArbiter:
    """Independent, authoritative arbitration between a plan and the actuators."""

    def __init__(self, limits: ArbiterLimits | None = None) -> None:
        self.limits = limits or ArbiterLimits()
        self._evidence = EvidenceBook(self.limits.evidence)
        self._shaper = JerkShaper(
            comfort_jerk_mps3=self.limits.comfort_jerk_mps3,
            emergency_jerk_mps3=self.limits.emergency_jerk_mps3,
            emergency_decel_mps2=self.limits.emergency_grade_mps2,
            release_rate_mps3=self.limits.release_jerk_mps3,
        )
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Clear every latch and every history. The only way out of DISENGAGE."""
        self._latched_state = SafetyState.NOMINAL
        self._health_fault_streak = 0
        self._clean_streak = 0
        self._dropout_streak = 0
        self._blind_frames = 0
        self._miss_frames = 0
        self._held_avoid_mps2 = 0.0
        self._prev_speed_mps: float | None = None
        self._prev_accel_mps2: float | None = None
        self._kinematics_streak: Dict[str, int] = {}
        self._prev_steering_rad = 0.0
        self._prev_request_rad = 0.0
        self._last_commanded_steering = 0.0
        self._prev_throttle = 0.0
        self._prev_brake = 0.0
        self._last_timestamp_s: float | None = None
        self._clock_s = 0.0
        self._capture_s = 0.0
        self._log_throttle = Throttle(self.limits.log_repeat_period_s)
        self._logged_state = SafetyState.NOMINAL
        self._logged_violations: Tuple[str, ...] = ()
        self.lane_offset_available = False
        self._evidence.reset()
        self._shaper.reset(0.0)
        # The only per-track state left in the hazard path, and neither entry
        # can change a command: both exist so that a CHANGE of range provenance
        # can be reported once instead of every frame.
        self._range_source: Dict[int, RangeSource] = {}
        self._stable_source: Dict[int, RangeSource] = {}
        # How many frames ago the arbiter last overrode a channel itself.
        self._long_override_age = 999
        self._lat_override_active = False
        self._in_path_seen = False
        self._tracks_seen = False
        self.total_violations = 0
        self.total_overrides = 0
        self.last_lead: LeadAssessment | None = None
        self.last_shaping: List[str] = []

    @property
    def state(self) -> SafetyState:
        """Current latched safety state."""
        return self._latched_state

    @property
    def dropout_frames(self) -> int:
        """Consecutive frames on which perception was reported unhealthy."""
        return self._dropout_streak

    @property
    def demand_mps2(self) -> float:
        """The arbiter's own jerk-shaped deceleration demand, m/s^2."""
        return self._shaper.decel_mps2

    # ------------------------------------------------------------------- main

    def arbitrate(
        self,
        plan: MotionPlan | None,
        command: ControlCommand,
        state: SafetyContext,
    ) -> ArbitrationResult:
        """Decide what the actuators actually receive.

        Args:
            plan: The planner's motion plan, or None when planning itself failed.
                None is treated as a planning dropout, not as "no constraint".
            command: The controller's proposed command.
            state: The :class:`SafetyContext` for this frame.

        Returns:
            An :class:`ArbitrationResult`. ``result.command`` is authoritative:
            the caller must send exactly that to the actuators and must not fall
            back to ``command`` under any circumstance.
            ``result.violations`` is ``faults + mitigated + advisory + hazards``;
            see the module docstring for what each list means and for which one
            -- only ``faults`` -- can latch ``DISENGAGE``.
        """
        ctx = state
        faults: List[str] = []
        mitigated: List[str] = []
        hazards: List[str] = []
        advisory: List[str] = []
        self.last_shaping = []

        dt_s = self._resolve_dt(ctx, faults)
        self._clock_s += dt_s
        # The abscissa every range fit is placed on.  Prefer the capture time the
        # caller reports; fall back to the frame timestamp, and finally to the
        # accumulated decision clock, so a caller that reports neither is no
        # worse off than before.
        capture_t = state.measurement_t_s
        if capture_t is None or not math.isfinite(capture_t):
            capture_t = state.timestamp_s
        if capture_t is None or not math.isfinite(capture_t):
            capture_t = self._clock_s
        self._capture_s = float(capture_t)
        ego_speed_mps, ego_ok = self._resolve_ego(ctx, faults, mitigated)
        cmd_in, cmd_ok = self._sanitize(command, faults, mitigated)

        if ctx.camera_calibrated is False and not self.limits.allow_uncalibrated_range:
            # Every metric range below is derived from this camera's geometry. An
            # uncalibrated camera makes them assumptions, so the arbiter says so
            # and refuses to report NOMINAL while acting on them.
            mitigated.append("camera_uncalibrated")

        perception = ctx.perception if ctx.perception is not None else PerceptionStatus()
        if perception.ok:
            self._dropout_streak = 0
            self._blind_frames = 0
        else:
            self._dropout_streak = max(
                self._dropout_streak + 1, int(perception.consecutive_failures)
            )
            self._blind_frames += 1
            faults.append(
                "perception_dropout_%d%s"
                % (self._dropout_streak, (":" + perception.reason) if perception.reason else "")
            )

        # The lead is assessed on EVERY frame, dropout included.  The tracker
        # coasts its tracks through a perception blink, and discarding the whole
        # assessment on those frames made the arbiter forget the hazard it was
        # braking for.  A coasted track is degraded, not absent: it is dropped
        # once ``time_since_update`` passes ``max_coast_frames``, and its
        # extrapolated range is never folded into the measured-rate window.
        lead = self._assess_lead(ctx, ego_speed_mps, ego_ok, dt_s, mitigated, advisory)
        self.last_lead = lead

        aeb, hazard = self._classify_hazard(lead, hazards)
        plan_missing = plan is None
        if plan_missing:
            faults.append("plan_missing")
        else:
            self._check_plan(plan, ego_speed_mps, ego_ok, faults, mitigated, advisory)
        self._check_kinematics(
            ego_speed_mps, ego_ok, dt_s, aeb or hazard, mitigated, advisory
        )

        self._check_lane_departure(ctx, mitigated)
        steering_out = self._limit_steering(
            cmd_in.steering, ego_speed_mps, dt_s, mitigated, advisory
        )

        requested = self._decide_state(faults, mitigated, hazard, cmd_ok, ego_ok, aeb)
        # Is a minimum-risk manoeuvre actually in progress?  It is while an
        # emergency test is tripped, while the vehicle cannot see, and while
        # the arbiter's own demand is still at emergency grade.
        manoeuvring = (
            aeb
            or plan_missing
            or not perception.ok
            or not ego_ok
            or not cmd_ok
            or self._shaper.decel_mps2 >= self.limits.emergency_grade_mps2
        )
        new_state = self._latch(requested, manoeuvring)

        out = self._synthesise_command(
            cmd_in,
            steering_out,
            new_state,
            aeb,
            hazard,
            lead,
            dt_s,
            cmd_ok,
            ego_ok,
            ego_speed_mps,
            bool(perception.ok),
            mitigated,
            plan_missing,
        )

        self._check_command_jerk(
            out.brake, dt_s, aeb or hazard, mitigated, new_state, bool(perception.ok)
        )

        violations = faults + mitigated + advisory + hazards
        if violations:
            self.total_violations += 1
        if new_state is not SafetyState.NOMINAL:
            self.total_overrides += 1

        self._prev_steering_rad = out.steering * self.limits.max_road_wheel_rad
        # Updated on EVERY frame, whatever the state. The previous code only wrote
        # this in NOMINAL, so an MRM entered in a bend held 0.0 -- it straightened
        # the wheel into the corner and then raised a steering_rate fault against
        # its own hold.
        self._last_commanded_steering = out.steering
        self._prev_throttle = out.throttle
        self._prev_brake = out.brake
        # Record what the arbiter itself did to the actuators this frame, so that
        # next frame's checks can tell the vehicle's response to the ARBITER apart
        # from the planner's or the world's behaviour.
        # "Did the arbiter drive the longitudinal channel this frame?" is a
        # question about the CHANNEL, not about the state label.  Answering it
        # with ``state is not NOMINAL`` made it self-fulfilling: any mitigated
        # finding -- including one about the vehicle's own measured deceleration
        # -- forces LIMITED, LIMITED then counted as an override, and the next
        # frame's identical finding was re-labelled "the arbiter caused this" and
        # excused.  A finding excused by a state that the finding itself created
        # is not an attribution, it is a loop.
        long_override = (
            out.brake > cmd_in.brake + 1e-3 or out.throttle < cmd_in.throttle - 1e-3
        )
        self._long_override_age = 0 if long_override else min(self._long_override_age + 1, 999)
        self._lat_override_active = (
            _SEVERITY[new_state] >= _SEVERITY[SafetyState.MIN_RISK_MANEUVER]
        )
        if ego_ok:
            self._prev_speed_mps = ego_speed_mps
        if ctx.timestamp_s is not None and math.isfinite(ctx.timestamp_s):
            self._last_timestamp_s = ctx.timestamp_s

        reason = self._describe(new_state, violations, lead, aeb)
        self._log_decision(new_state, out, violations, reason)
        return ArbitrationResult(
            command=out, state=new_state, violations=list(violations), reason=reason
        )

    # ------------------------------------------------------------- input prep

    def _log_decision(
        self,
        new_state: SafetyState,
        out: ControlCommand,
        violations: List[str],
        reason: str,
    ) -> None:
        """Emit at most one line per CHANGE, plus a throttled heartbeat.

        Three cases, and the distinction is the whole point:

        * **a transition** -- the state changed, or the KIND of violation changed
          (see :func:`_condition_key`, which compares violations with their
          numbers normalised away) -- is logged at WARNING on the frame it
          happens, never throttled.  An entry, an exit and a change of cause are
          the events an operator acts on, and there is one per transition rather
          than one per frame;
        * **a sustained non-nominal state** with the same violations is logged at
          most once per ``limits.log_repeat_period_s``, carrying the number of
          frames suppressed since the last line;
        * **NOMINAL with nothing to say** is silent.

        Returning to NOMINAL logs once at INFO, so a run's log shows the
        recovery and not just the fault.  Nothing here changes a command; a
        logging failure cannot affect the actuators.
        """
        current = _condition_key(violations)
        changed = new_state is not self._logged_state or current != self._logged_violations

        if new_state is SafetyState.NOMINAL:
            if changed and self._logged_state is not SafetyState.NOMINAL:
                logger.info(
                    "Arbiter recovered to nominal: cmd t=%.2f b=%.2f s=%.2f | %s",
                    out.throttle, out.brake, out.steering, reason,
                )
            self._logged_state = new_state
            self._logged_violations = current
            return

        if changed:
            self._log_throttle = Throttle(self.limits.log_repeat_period_s)
            self._log_throttle.ready()  # consume the first slot for this line
            logger.warning(
                "Arbiter %s: cmd t=%.2f b=%.2f s=%.2f | %s",
                new_state.value, out.throttle, out.brake, out.steering, reason,
            )
        elif self._log_throttle.ready():
            logger.warning(
                "Arbiter %s (sustained, %d frame(s) suppressed): "
                "cmd t=%.2f b=%.2f s=%.2f | %s",
                new_state.value, self._log_throttle.last_suppressed,
                out.throttle, out.brake, out.steering, reason,
            )
        self._logged_state = new_state
        self._logged_violations = current

    def _resolve_dt(self, ctx: SafetyContext, faults: List[str]) -> float:
        """Prefer the measured frame interval; clamp and flag implausible ones.

        A frame FASTER than ``min_dt_s`` is not a finding of any kind. Running well
        is not a fault: the previous version reported ``timing_dt_out_of_range`` for
        an unpaced 2 ms frame and fed it to the DISENGAGE counter, which latched the
        terminal state after 40 frames of an empty road.
        """
        lim = self.limits
        dt = ctx.dt_s if (isinstance(ctx.dt_s, (int, float)) and math.isfinite(ctx.dt_s)) else 0.0

        if ctx.timestamp_s is not None and math.isfinite(ctx.timestamp_s):
            if self._last_timestamp_s is not None:
                measured = ctx.timestamp_s - self._last_timestamp_s
                if measured <= 0.0:
                    faults.append("timing_non_monotonic_timestamp")
                elif measured > lim.max_frame_gap_s:
                    faults.append("timing_stale_input_%.3fs" % measured)
                    dt = lim.max_dt_s
                else:
                    dt = measured

        if dt <= 0.0 or not math.isfinite(dt):
            faults.append("timing_invalid_dt")
            return NOMINAL_DT_S
        if dt > lim.max_dt_s:
            # The loop is running slower than the arbiter's maths is valid for.
            # That is a health fault: the hazard estimates below are stale.
            faults.append("timing_dt_above_max_%.3fs" % dt)
            return lim.max_dt_s
        if dt < lim.min_dt_s:
            # Silently clamp. Nothing is wrong with a fast frame; the clamp only
            # keeps the per-frame rate windows from collapsing to zero.
            return lim.min_dt_s
        return dt

    def _resolve_ego(
        self, ctx: SafetyContext, faults: List[str], mitigated: List[str]
    ) -> Tuple[float, bool]:
        """Return ``(speed_mps, usable)``. An unusable ego state forces an MRM."""
        ego = ctx.ego
        if ego is None:
            faults.append("ego_state_missing")
            return 0.0, False
        if not ego.valid:
            faults.append("ego_state_invalid")
            return 0.0, False
        if not math.isfinite(ego.speed_mps) or ego.speed_mps < 0.0:
            faults.append("ego_speed_implausible")
            return 0.0, False
        if ego.speed_mps > self.limits.max_speed_mps * 1.5:
            # Reported and it degrades the state, but the ego state is still
            # USABLE, so this is not a health fault: the arbiter can and does keep
            # arbitrating on it.
            mitigated.append("ego_speed_above_envelope_%.1f" % ego.speed_mps)
            return ego.speed_mps, True
        return float(ego.speed_mps), True

    def _sanitize(
        self, cmd: ControlCommand, faults: List[str], mitigated: List[str]
    ) -> Tuple[ControlCommand, bool]:
        """Range- and NaN-check the proposed command.

        A non-finite field makes the whole command unusable; the arbiter reports
        that and hands back zeros, then escalates to a minimum-risk manoeuvre.  It
        does NOT clamp NaN, because ``max(0, min(1, nan))`` evaluates to 1.0 in
        Python and would turn a corrupt command into full braking.
        """
        values = (cmd.throttle, cmd.brake, cmd.steering)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            faults.append("command_not_finite")
            logger.error("Non-finite control command received: %r", cmd)
            return ControlCommand(throttle=0.0, brake=0.0, steering=0.0), False

        throttle, brake, steering = (float(v) for v in values)
        if not (0.0 <= throttle <= 1.0):
            mitigated.append("command_throttle_out_of_range_%.3f" % throttle)
            throttle = _clamp(throttle, 0.0, 1.0)
        if not (0.0 <= brake <= 1.0):
            mitigated.append("command_brake_out_of_range_%.3f" % brake)
            brake = _clamp(brake, 0.0, 1.0)
        if not (-1.0 <= steering <= 1.0):
            mitigated.append("command_steering_out_of_range_%.3f" % steering)
            steering = _clamp(steering, -1.0, 1.0)
        if throttle > 0.02 and brake > 0.02:
            mitigated.append("command_throttle_brake_conflict")
            throttle = 0.0
        return ControlCommand(throttle=throttle, brake=brake, steering=steering), True

    # ------------------------------------------------------------ lead + risk

    def _corridor_anchors(self, ctx: SafetyContext) -> Tuple[List[float], float]:
        """Return ``(anchor_columns_px, half_width_px)`` for the image corridor.

        The image centre is ALWAYS an anchor: it is where the ego is going, and it
        is the one reference column that no perception module can move.  A lane
        model adds a SECOND anchor -- widening the corridor -- but only when it is
        real and confident.  A lane centre can therefore never narrow the corridor
        or shift it off the ego's own heading.
        """
        lim = self.limits
        frac = max(float(ctx.ego_lane_half_width_frac or 0.0), lim.min_in_path_half_width_frac)
        half_px = ctx.frame_width_px * frac
        anchors = [ctx.frame_width_px / 2.0]
        lane = ctx.lane
        if (
            lane is not None
            and not lane.is_mock
            and math.isfinite(lane.lane_center_px)
            and 0.0 <= lane.lane_center_px <= ctx.frame_width_px
            and lane.confidence >= lim.lane_trust_confidence
        ):
            anchors.append(float(lane.lane_center_px))
        return anchors, half_px

    def _object_half_width_m(self, ctx: SafetyContext, obj: TrackedObject) -> float:
        """Half the object's width in metres, from its own box and range.

        The box subtends ``2 w f / d`` pixels for a half width ``w`` at range
        ``d``, so the width follows from the measurement once a focal length is
        known.  ``SafetyContext.camera_focal_px`` carries the calibrated one; in
        its absence the arbiter uses the focal length implied by a 60 deg
        horizontal field of view, which is the narrowest lens a forward ADAS
        camera is built with and therefore the one that yields the SMALLEST
        estimate for a given box -- and the estimate is then floored at half a
        passenger car, so the corridor can only ever be widened by this
        calculation, never narrowed.  Widening is the safe direction: a wider
        object overlaps the ego's path from further away.
        """
        floor = self.limits.ego_half_width_m
        box = getattr(obj, "box", None)
        if box is None or not math.isfinite(obj.distance_m) or obj.distance_m <= 0.0:
            return floor
        try:
            width_px = abs(float(box.x2) - float(box.x1))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return floor
        focal = getattr(ctx, "camera_focal_px", None)
        if not (focal and math.isfinite(focal) and focal > 0.0):
            if ctx.frame_width_px <= 0:
                return floor
            # f = (W/2) / tan(hfov/2), hfov = 60 deg.
            focal = (ctx.frame_width_px / 2.0) / math.tan(math.radians(30.0))
        return max(floor, 0.5 * width_px * float(obj.distance_m) / focal)

    def _in_path(self, ctx: SafetyContext, obj: TrackedObject) -> bool:
        """The arbiter's OWN in-path test, in metres first and pixels second.

        It does not read ``TrackedObject.in_ego_lane``.  That flag is set by the
        tracker from the same lane model the planner reads, so a lane-detection
        error would reach the planner and the "independent" backstop identically
        -- the definition of a common-mode failure.  Worse, it fails in BOTH
        directions: a lane fit that has slipped toward the kerb hides a car that
        is directly in front of the bumper, and one that has slipped the other way
        reports a car in the next lane as being in the way.  A 0.86 m lane error
        is enough to drag a 1.8 m car at 3.5 m inside the *reported* ego lane.

        What decides whether a collision is geometrically possible is the object's
        lateral offset FROM THE EGO, which the detector measures from the box's
        position in the image and which no lane model can move.  Half the ego plus
        half the object is 1.8 m for two passenger cars; a car one lane over is at
        3.5 m and is not in the way however confident the lane fit is.  The
        arbiter adds ``lateral_gate_margin_m`` on top, so its corridor is strictly
        WIDER than the primary path's.

        When no metric lateral offset is available the test falls back to the
        image corridor, where an object counts as in path when ANY part of its box
        overlaps -- it is far better to consider an out-of-lane object than to
        miss a real lead.
        """
        lim = self.limits
        lateral = getattr(obj, "lateral_offset_m", None)
        if lateral is not None and math.isfinite(lateral) and obj.distance_m > 0.0:
            half_w = self._object_half_width_m(ctx, obj)
            return abs(float(lateral)) <= lim.ego_half_width_m + half_w + lim.lateral_gate_margin_m

        if ctx.frame_width_px <= 0:
            return True
        anchors, half_px = self._corridor_anchors(ctx)
        if half_px <= 0.0:
            return True
        left = min(obj.box.x1, obj.box.x2)
        right = max(obj.box.x1, obj.box.x2)
        if not (math.isfinite(left) and math.isfinite(right)):
            return True
        for anchor in anchors:
            if right >= anchor - half_px and left <= anchor + half_px:
                return True
        return False

    def _fuse_range(
        self,
        ctx: SafetyContext,
        obj: TrackedObject,
        mitigated: List[str],
        advisory: List[str],
    ) -> Tuple[float, RangeSource, bool]:
        """The NEARER of two range channels that CORROBORATE each other.

        Returns ``(distance_m, source, disagreement)``.  A pure function of THIS
        frame's two measurements: it holds no state, so no sequence of inputs can
        put it in a mode, and it cannot latch.

        The rule is two lines:

        * the channels agree within ``range_disagreement_frac`` -- use the NEARER
          of the two, because a backstop must never be talked out of the shorter
          of two ranges it has been given and both are credible;
        * they do not -- use the pinhole, report
          ``range_channel_disagreement_...`` and let it degrade the state.  Two
          ranges 12x apart are not two opinions about one distance; at least one
          channel is broken, and the pinhole is the one with a geometric
          derivation and a calibration behind it.  Degrading is the response to a
          channel fault; silently adopting the broken channel is not.

        Args:
            ctx: The frame's context; ``ctx.independent_ranges`` is consulted
                before ``obj.range_estimate``.
            obj: The track whose ``distance_m`` is the pinhole range, metres.
            mitigated: Findings that degrade the state are appended here.
            advisory: Findings that are reported but do not degrade the state.

        Failure behaviour:
            An absent, ``UNAVAILABLE``, non-finite or low-confidence second
            channel is discarded and the pinhole range is returned with
            ``RangeSource.PINHOLE``, so a consumer can always tell whether the
            cross-check ran.  A NaN confidence is discarded explicitly: ``nan <
            0.35`` is False, so an unguarded comparison used to pass a NaN
            straight into the fused range, and every hazard test against NaN is
            False -- a real obstacle produced no finding at all.

        This replaces a 131-line state machine (a confidence gate with
        hysteresis, an asymmetric adoption/dwell counter, a disagreement
        corroboration streak and a confidence-weighted blend) carrying five
        per-track dictionaries.  Two of its parts were provably inert and the
        third was a latch:

        * the confidence-weighted blend returned a value strictly between the two
          channels and the very next expression took ``min`` of the blend and
          both channels -- which is ``min`` of the two channels for any weight.
          131 lines whose central arithmetic was a no-op;
        * the hysteresis and the dwell counter existed to stop the reported
          PROVENANCE flip-flopping (26 times in 400 frames of real footage).
          That is a logging problem and it is now solved in the logger, which
          rate-limits a repeated condition instead of asking the decision to be
          sticky;
        * the corroboration streak delayed adopting a persistently disagreeing
          NEARER channel by three frames and then adopted it anyway.  A channel
          that disagrees by more than a third does not become correct by
          repeating itself, and the response to a broken range channel is to
          report it and degrade -- which happens on the first frame now, instead
          of the fourth.
        """
        lim = self.limits
        tid = obj.track_id
        pinhole = obj.distance_m

        estimate: RangeEstimate | None = None
        if ctx.independent_ranges:
            estimate = ctx.independent_ranges.get(tid)
        if estimate is None:
            estimate = obj.range_estimate

        if (
            estimate is None
            or estimate.source is RangeSource.UNAVAILABLE
            or not math.isfinite(estimate.distance_m)
            or estimate.distance_m < 0.0
        ):
            return pinhole, RangeSource.PINHOLE, False

        if not math.isfinite(estimate.confidence):
            mitigated.append("range_channel_confidence_not_finite_track%d" % tid)
            return pinhole, RangeSource.PINHOLE, False

        if estimate.confidence < lim.min_range_confidence:
            mitigated.append(
                "range_channel_low_confidence_%.2f_track%d" % (estimate.confidence, tid)
            )
            return pinhole, RangeSource.PINHOLE, False

        reference = max(1.0, min(pinhole, estimate.distance_m))
        relative = abs(pinhole - estimate.distance_m) / reference
        if relative > lim.range_disagreement_frac:
            mitigated.append(
                "range_channel_disagreement_%.0f%%_%s_track%d"
                % (
                    relative * 100.0,
                    "farther" if estimate.distance_m >= pinhole else "nearer",
                    tid,
                )
            )
            return pinhole, RangeSource.PINHOLE, True

        if estimate.distance_m >= pinhole:
            return pinhole, RangeSource.PINHOLE, False
        return estimate.distance_m, RangeSource.FUSED, False

    def _rss_gap_m(self, ego_speed_mps: float, lead_speed_mps: float) -> float:
        """RSS-style minimum safe gap for a given pair of speeds, metres."""
        lim = self.limits
        rss = (
            ego_speed_mps * lim.reaction_time_s
            + (ego_speed_mps * ego_speed_mps) / (2.0 * lim.ego_brake_capability_mps2)
            - (lead_speed_mps * lead_speed_mps) / (2.0 * lim.lead_brake_capability_mps2)
        )
        return max(lim.absolute_min_gap_m, rss)

    def _assess_lead(
        self,
        ctx: SafetyContext,
        ego_speed_mps: float,
        ego_ok: bool,
        dt_s: float,
        mitigated: List[str],
        advisory: List[str],
    ) -> LeadAssessment | None:
        """Select and characterise the most dangerous in-path object.

        Selection is by the deceleration the object actually REQUIRES, falling
        back to smallest range when nothing is measurably closing -- not by
        ``min(distance)``, which is the rule the planner and the old pipeline both
        used and which selects a slower car in the next lane over the braking one
        in this lane.

        There is exactly one closing rate here and it is a measurement.  A track
        the arbiter has not seen four distinct captures of reports
        ``rate_is_measured = False`` and contributes NO braking; the previous
        revision seeded such a track at ``-ego_speed`` and let the emergency tests
        read the seed, which is how a lead at a constant 32.5 m brought a 20 m/s
        ego to a standstill.
        """
        lim = self.limits
        alive = [obj.track_id for obj in ctx.tracks]
        self._evidence.forget_all_but(alive)
        alive_set = set(alive)
        for book in (self._range_source, self._stable_source):
            for key in [k for k in book if k not in alive_set]:
                del book[key]

        a_ego = self._evidence.ego_accel_mps2(self._capture_s, ego_speed_mps) if ego_ok else 0.0

        best: LeadAssessment | None = None
        seen_in_path = False
        for obj in ctx.tracks:
            if not self._in_path(ctx, obj):
                continue
            seen_in_path = True
            if not math.isfinite(obj.distance_m) or obj.distance_m < 0.0:
                mitigated.append("track%d_range_implausible" % obj.track_id)
                continue
            coast_frames = int(getattr(obj, "time_since_update", 0) or 0)
            if coast_frames > lim.max_coast_frames:
                mitigated.append(
                    "track%d_coast_%dframes_too_stale" % (obj.track_id, coast_frames)
                )
                continue

            distance_m, source, disagreement = self._fuse_range(
                ctx, obj, mitigated, advisory
            )
            self._note_source(obj.track_id, source, coast_frames, advisory)

            evidence = self._evidence.track(obj.track_id)
            evidence.update(
                self._capture_s,
                distance_m,
                measured=(coast_frames == 0),
                capture_token=int(getattr(obj, "hits", 0) or 0) or None,
            )
            closure = evidence.closure(a_ego)
            if closure.reanchored:
                mitigated.append("range_jump_track%d" % obj.track_id)

            required = self._avoidance_decel(distance_m, closure, ego_speed_mps)
            closing = closure.closing_mps if closure.measured else 0.0
            # Time to contact is measured against the clearance a completed stop
            # preserves, and it exists only when there is a closure to divide by.
            # Dividing a gap by a rate that is a fraction of a metre per second
            # produces a "time to collision" for a vehicle inching forward in a
            # queue, and the standstill gap made it worse: subtracting 4 m from a
            # 2.4 m range gives a negative gap and a TTC of exactly zero for a
            # vehicle that has already stopped safely.
            gap = distance_m - lim.warrant_clearance_m
            if not closure.measured or closing <= lim.ttc_min_closing_mps:
                ttc_s = float("inf")
            elif gap <= 0.0:
                ttc_s = 0.0
            else:
                ttc_s = gap / closing

            rss = self._rss_gap_m(ego_speed_mps, max(0.0, ego_speed_mps - closing))
            rss_matched = self._rss_gap_m(ego_speed_mps, ego_speed_mps)

            candidate = LeadAssessment(
                track_id=obj.track_id,
                distance_m=distance_m,
                range_rate_mps=-closing,
                ttc_s=ttc_s,
                required_decel_mps2=required,
                rss_min_gap_m=rss,
                source=source,
                rss_matched_gap_m=rss_matched,
                disagreement=disagreement,
                reinitialised=closure.reanchored,
                coasting=coast_frames > 0,
                coast_frames=coast_frames,
                rate_is_measured=closure.measured,
                measured_rate_mps=(-closing if closure.measured else None),
                rate_updates=closure.samples,
                closing_lcb_mps=closure.closing_lcb_mps,
                closing_stderr_mps=closure.stderr_mps,
                lead_accel_mps2=closure.lead_accel_mps2,
                lateral_offset_m=getattr(obj, "lateral_offset_m", None),
            )
            if best is None or self._more_dangerous(candidate, best):
                best = candidate

        self._in_path_seen = seen_in_path
        self._tracks_seen = bool(ctx.tracks)
        return best

    def _note_source(
        self,
        track_id: int,
        source: RangeSource,
        coast_frames: int,
        advisory: List[str],
    ) -> None:
        """Record a change of range provenance, without calling it a degradation.

        Only a change between two MEASURED frames is a change of provenance worth
        reporting.  On real footage every single one of 26 reported "source
        switches" was a coast frame with no depth estimate followed by the depth
        estimate coming straight back, so a coast frame is skipped outright.

        Advisory only: this never degrades the state and never changes a command.
        The arbiter's own estimator changing its mind about provenance is not
        evidence that the automation is degraded.  Repetition is handled by the
        logger's rate limiter, not by making the decision sticky.
        """
        self._range_source[track_id] = source
        if coast_frames != 0:
            return
        stable = self._stable_source.get(track_id)
        if stable is not None and stable is not source:
            # The arbiter's own estimator changing its mind about provenance is
            # not evidence that the automation is degraded.
            advisory.append(
                "range_source_switch_%s_to_%s_track%d" % (stable.value, source.value, track_id)
            )
        self._stable_source[track_id] = source

    def _avoidance_decel(
        self, distance_m: float, closure: ClosureEstimate, ego_speed_mps: float
    ) -> float:
        """The deceleration this object requires, m/s^2.  The evidence gate.

        Zero unless a closure has been MEASURED and its
        :attr:`~adas.control.evidence.EvidenceLimits.closure_sigmas` lower
        confidence bound is clear of zero.  **The gate is on the bound; the
        magnitude is the estimate**: once the gate is open the requirement is
        computed from the unbiased closure, because it goes as the square of that
        number and braking for a deliberately pessimistic closure would be its own
        kind of over-response.

        The lead-acceleration credit is applied whether or not the closure gate is
        open, because a lead that is measurably stopping is a hazard even at a gap
        that is not yet shrinking -- but it is itself gated, at a higher
        confidence, inside :meth:`RangeEvidence.closure`.
        """
        lim = self.limits
        if not closure.measured:
            return 0.0
        gated_closing = closure.closing_mps if closure.confident_closing else 0.0
        if gated_closing <= 0.05 and closure.lead_accel_mps2 >= -0.5:
            return 0.0
        required = required_decel_mps2(
            range_m=distance_m,
            closing_mps=gated_closing,
            ego_speed_mps=ego_speed_mps,
            lead_accel_mps2=closure.lead_accel_mps2,
            target_clearance_m=lim.target_clearance_m,
            current_decel_mps2=self._shaper.decel_mps2,
            max_decel_mps2=lim.max_deceleration_mps2,
            jerk_mps3=lim.emergency_jerk_mps3,
        )
        if required <= 0.05:
            return 0.0
        demand = min(
            lim.max_deceleration_mps2,
            required * (1.0 + lim.demand_margin_frac) + lim.demand_margin_mps2,
        )
        # The sub-emergency band is reserved for a warranted emergency; see
        # :func:`adas.control.evidence.sub_emergency_guard`.  The warrant is the
        # arbiter's OWN causal requirement -- what constant deceleration would
        # just hold the standstill clearance, with no margin and no ramp
        # allowance -- so the guard opens on the physics rather than on the
        # arbiter's own inflated demand.
        warrant = required_decel_mps2(
            range_m=distance_m,
            closing_mps=gated_closing,
            ego_speed_mps=ego_speed_mps,
            lead_accel_mps2=closure.lead_accel_mps2,
            target_clearance_m=lim.warrant_clearance_m,
            current_decel_mps2=0.0,
            max_decel_mps2=lim.max_deceleration_mps2,
            jerk_mps3=0.0,
        )
        return sub_emergency_guard(
            demand, warrant, lim.comfort_decel_mps2, lim.band_guard_margin_mps2
        )

    @staticmethod
    def _more_dangerous(a: LeadAssessment, b: LeadAssessment) -> bool:
        """Order two candidates: larger required deceleration first, then nearer.

        Range alone is the wrong key and it is the one the old pipeline used: a
        distractor 18 m away in the next lane is nearer than the car at 26 m in
        this one that is braking at 6 m/s^2, and picking the nearer selects the
        object that is not going to be hit.
        """
        if a.required_decel_mps2 != b.required_decel_mps2:
            return a.required_decel_mps2 > b.required_decel_mps2
        if a.ttc_s != b.ttc_s:
            return a.ttc_s < b.ttc_s
        return a.distance_m < b.distance_m

    def _classify_hazard(
        self, lead: LeadAssessment | None, hazards: List[str]
    ) -> Tuple[bool, bool]:
        """Return ``(aeb, hazard)`` and append the findings that justify them.

        These describe traffic, not system health, so none of them may latch
        DISENGAGE.  ``aeb`` requests MIN_RISK_MANEUVER; ``hazard`` requests
        LIMITED.

        Every emergency test here reads a MEASURED closure.  There is deliberately
        no rate-independent emergency: the two the previous revision carried -- a
        gap below the absolute minimum, and a gap inside half the speed-matched
        RSS distance -- fire on a lead that is holding station perfectly, because
        they are tests on the RANGE and a short range at a matched speed is a
        headway problem and not a collision.  They are reported here, and they
        floor the state at LIMITED so the condition is visible and the throttle is
        locked out, but they authorise no braking: closing a short gap is the
        following law's job and it does it at 1.5 m/s^2.
        """
        if lead is None:
            return False, False
        lim = self.limits
        hazard = False

        # -- RANGE-ONLY tests.  These need no closing rate at all: they are
        # computed from the measured range and the measured ego speed, so they
        # are available on a track's very first frame.  What they support is one
        # rung of authority -- a degraded state, and with it the throttle lockout
        # -- and no braking, because a short gap at a matched speed is a headway
        # problem and closing it is the following law's job.
        if lead.distance_m < lim.absolute_min_gap_m:
            hazards.append("gap_%.2fm_below_absolute_min" % lead.distance_m)
            hazard = True
        if lead.distance_m < lim.aeb_headway_frac * lead.rss_matched_gap_m:
            hazards.append(
                "headway_%.1fm_below_%.0f%%_of_matched_rss_%.1fm"
                % (lead.distance_m, lim.aeb_headway_frac * 100.0, lead.rss_matched_gap_m)
            )
            hazard = True

        if not lead.rate_is_measured:
            # Everything below reads a closure, and there is not one yet.  Silence
            # is the correct output for those: an "unmeasured rate" finding would
            # floor the state at LIMITED for the first frames of every track the
            # system ever sees, including on an empty motorway where a car merely
            # appeared at 90 m.
            return False, hazard

        if lead.distance_m < lead.rss_min_gap_m:
            hazards.append(
                "headway_%.1fm_below_rss_%.1fm" % (lead.distance_m, lead.rss_min_gap_m)
            )

        # THE EMERGENCY TEST IS THE REQUIRED DECELERATION, AND NOTHING ELSE.
        # Time to collision and required deceleration are the same measurement
        # expressed two ways, and only one of them is actionable: a system
        # decides how hard to brake, not how long it has.  Worse, they disagree
        # in the one place it matters.  A vehicle that has almost completed a
        # correct stop sits a couple of metres from the obstacle still creeping
        # at half a metre a second: the time to contact is short, the
        # deceleration required is 0.15 m/s^2, and a TTC trigger declares a
        # minimum-risk manoeuvre for a car-park creep -- which the harness
        # correctly calls ``unwarranted_authority_state``.  TTC stays as a
        # reported hazard, because it is a true and useful statement about the
        # traffic; it does not latch a manoeuvre.
        emergency: List[str] = []
        if lead.required_decel_mps2 >= lim.aeb_required_decel_mps2:
            emergency.append(
                "required_decel_%.1f_above_%.1f"
                % (lead.required_decel_mps2, lim.aeb_required_decel_mps2)
            )
        if lead.ttc_s < lim.ttc_brake_s:
            hazards.append("ttc_%.2fs_below_brake_threshold" % lead.ttc_s)
            hazard = True
        elif lead.ttc_s < lim.ttc_warn_s:
            hazards.append("ttc_%.2fs_below_warn_threshold" % lead.ttc_s)
            hazard = True
        if lead.required_decel_mps2 >= lim.comfort_decel_mps2:
            hazard = True

        # LATCH-FREE.  The manoeuvre is requested on the frame the requirement is
        # there and dropped on the frame it is not; ``aeb`` is a pure function of
        # this frame's measurements, exactly like ``lead.required_decel_mps2``
        # that produced it.  There is no frame counter, because a counter is a
        # latch -- state that outlives the measurement that set it -- and every
        # self-sustaining loop in this module's history was a latch holding on
        # evidence the system was itself producing.  What a counter was there to
        # reject, a single noisy frame, is rejected upstream and harder: the
        # closure must clear four standard errors of zero and the lead
        # deceleration five before either reaches this function at all.
        aeb = bool(emergency)
        if emergency:
            hazards.extend(emergency)
            hazard = True
        return aeb, hazard

    # ------------------------------------------------------------- envelope

    def _self_induced(self, frames: int = 2) -> bool:
        """Was the ARBITER itself driving the longitudinal channel recently?

        The general principle this implements: **the arbiter must not treat a
        consequence of its own intervention as evidence that the intervention is
        still needed.**  An arbiter that brakes, measures the resulting
        deceleration or the resulting gap between the planner's target speed and
        the ego speed, calls that a violation, and keeps braking because of it,
        has built a latch out of its own actuator.

        ``frames`` is how long the arbiter's own action can still EXPLAIN the
        measurement in question, and it differs by finding.  An instantaneous
        derivative -- achieved deceleration, jerk -- is explained by the last frame
        or two.  A persistent state variable -- the gap between the planner's
        target speed and the measured ego speed -- is opened by the intervention
        and does not close again until the vehicle has been allowed to accelerate
        for a while, so that one uses ``recovery_frames``.

        Findings that fail this test are still reported in ``violations`` and in
        the logs.  They just may not force a state or reset the recovery streak.
        """
        return self._long_override_age <= frames

    def _check_plan(
        self,
        plan: MotionPlan,
        ego_speed_mps: float,
        ego_ok: bool,
        faults: List[str],
        mitigated: List[str],
        advisory: List[str],
    ) -> None:
        """Envelope checks on the plan itself.

        A NON-FINITE plan field is a health fault: the planner is producing
        garbage. A plan that is merely outside the envelope is mitigated -- the
        arbiter clamps the resulting command and continues.
        """
        lim = self.limits
        if not math.isfinite(plan.target_speed_mps) or plan.target_speed_mps < 0.0:
            faults.append("plan_target_speed_invalid")
        elif plan.target_speed_mps > lim.max_speed_mps:
            mitigated.append(
                "plan_speed_%.1f_above_%.1f" % (plan.target_speed_mps, lim.max_speed_mps)
            )
        if not math.isfinite(plan.steering_angle_deg):
            faults.append("plan_steering_invalid")
        elif abs(math.radians(plan.steering_angle_deg)) > lim.max_steering_angle_rad:
            mitigated.append("plan_steering_%.1fdeg_above_limit" % plan.steering_angle_deg)

        # There is deliberately NO check here on
        # ``(target_speed - ego_speed) / plan_horizon_s``.  A plan's target speed
        # is a SETPOINT, not a trajectory: "resume 20 m/s" from 16.7 m/s is a
        # perfectly ordinary cruise request, and dividing the difference by a
        # fictitious one-second horizon turns it into a 3.3 m/s^2 "requested
        # acceleration" that no part of the system ever intends to produce -- the
        # controller's own rate limits see to that.  Reporting it degraded the
        # state for the whole of every recovery, which reset the recovery streak,
        # which held the state degraded: a latch built out of the arbiter's own
        # intervention, in the same family as the two that preceded it.
        #
        # The quantity that matters is the acceleration the vehicle ACHIEVES, and
        # that is measured directly in :meth:`_check_kinematics` from the vehicle
        # bus.  A setpoint is checked for being inside the speed envelope, which
        # is what the branch above does, and for nothing else.

    def _check_kinematics(
        self,
        ego_speed_mps: float,
        ego_ok: bool,
        dt_s: float,
        hazard_active: bool,
        mitigated: List[str],
        advisory: List[str],
    ) -> None:
        """Judge the ACHIEVED longitudinal acceleration and jerk.

        Two changes from the naive version, both of which are about not indicting
        the vehicle for the estimator's own noise:

        * the acceleration is a LEAST-SQUARES SLOPE of the measured speed over the
          evidence window, not a one-frame difference quotient.  Differentiating a
          noisy wheel speed frame by frame turns +/-0.05 m/s of bus noise into
          +/-2 m/s^2 and +/-40 m/s^3, which is larger than every limit here and is
          a statement about the sensor rather than about the control;
        * an exceedance must persist for ``kinematics_streak_frames`` before it is
          reported.  A real over-acceleration lasts; a draw from the noise does
          not.

        These are MEASUREMENTS of the vehicle's response, reported and used to
        degrade the state, never health faults: the arbiter cannot un-apply an
        acceleration that already happened.
        """
        if not ego_ok:
            return
        lim = self.limits
        if self._prev_speed_mps is None:
            self._prev_accel_mps2 = None
            return
        if max(ego_speed_mps, self._prev_speed_mps) < lim.kinematics_min_speed_mps:
            # See ArbiterLimits.kinematics_min_speed_mps.
            self._prev_accel_mps2 = None
            self._kinematics_streak.clear()
            return
        # The arbiter can only ever REMOVE energy, so an excess ACCELERATION is
        # never its own doing and stays a real finding. An excess DECELERATION or a
        # jerk spike very often is: they are the vehicle answering the arbiter's own
        # brake command.
        induced = self._self_induced()
        # The SAME abscissa the range window uses: EvidenceBook keeps one ego
        # speed history and two callers stamping it from two different clocks
        # would interleave two timelines into one least-squares fit.
        accel = self._evidence.ego_accel_mps2(self._capture_s, ego_speed_mps)

        if self._sustained("accel", accel > lim.max_acceleration_mps2 + 1e-6):
            mitigated.append("measured_accel_%.2f_above_%.2f" % (accel, lim.max_acceleration_mps2))
        if self._sustained(
            "decel", -accel > lim.max_deceleration_mps2 + 1e-6 and not hazard_active
        ):
            text = "measured_decel_%.2f_above_%.2f" % (-accel, lim.max_deceleration_mps2)
            if induced:
                advisory.append(text + "_arbiter_commanded")
            else:
                mitigated.append(text + "_unjustified")
        self._prev_accel_mps2 = accel

    def _check_command_jerk(
        self,
        brake_out: float,
        dt_s: float,
        emergency: bool,
        mitigated: List[str],
        state: SafetyState = SafetyState.NOMINAL,
        perception_ok: bool = True,
    ) -> None:
        """Judge the jerk of the command the arbiter is about to ACTUATE.

        The previous version differentiated the measured ego speed twice and
        reported the result.  That is the wrong signal in two independent ways.
        It is a statement about the VEHICLE and the wheel-speed sensor rather than
        about the system's output -- +/-0.05 m/s of bus noise is +/-40 m/s^3 of
        apparent jerk, larger than every limit here -- and it is unactionable,
        because the arbiter cannot un-apply an acceleration that has already
        happened.  Worse, on a vehicle answering a perfectly legal 20 m/s^3
        demand through a 0.15 s brake rise it reports the vehicle for obeying.

        What the specification actually bounds, and what this component actually
        controls, is the rate of change of the DEMAND.  So that is what is
        checked: the arbiter's own output against its own previous output, at the
        comfort ceiling outside a collision-avoidance manoeuvre and the emergency
        ceiling inside one.  By construction the arbiter's own demand is shaped to
        those ceilings, so this can only fire on a step that arrived from upstream
        -- which is exactly the defect worth reporting, and is a statement about
        the bytes the primary path emitted.
        """
        lim = self.limits
        if dt_s <= 0.0:
            return
        decel = brake_out * lim.brake_authority_mps2
        prev = self._prev_brake * lim.brake_authority_mps2
        if decel <= prev:
            # Only a RISING demand is a comfort hazard: a release returns the
            # occupant toward zero g and is arrested by the seat back, and every
            # other requirement here demands that an unwarranted deceleration be
            # removed promptly.
            return
        # The band follows the manoeuvre, not the number.  A minimum-risk stop
        # and a collision-avoidance stop are both manoeuvres the emergency band
        # exists for; a comfort-limited ramp is not.  Judging an MRM's build-up
        # against the comfort ceiling would report the vehicle for making the
        # controlled stop the specification requires of it when it goes blind.
        urgent = (
            emergency
            or not perception_ok
            or _SEVERITY[state] >= _SEVERITY[SafetyState.MIN_RISK_MANEUVER]
            or max(decel, prev) >= lim.emergency_grade_mps2
        )
        limit = lim.emergency_jerk_mps3 if urgent else lim.comfort_jerk_mps3
        jerk = (decel - prev) / dt_s
        if jerk > limit + 1e-6:
            mitigated.append("jerk_%.1f_above_%.1f" % (jerk, limit))

    def _sustained(self, key: str, tripped: bool) -> bool:
        """True once ``key`` has been tripped ``kinematics_streak_frames`` in a row."""
        if not tripped:
            self._kinematics_streak[key] = 0
            return False
        streak = self._kinematics_streak.get(key, 0) + 1
        self._kinematics_streak[key] = streak
        return streak >= self.limits.kinematics_streak_frames

    def _check_lane_departure(self, ctx: SafetyContext, mitigated: List[str]) -> None:
        """Enforce ``max_lateral_offset_m`` when a metric lane offset is available.

        Sets :attr:`lane_offset_available` so the caller and the logs can tell the
        difference between "inside the lane" and "we could not tell". This is the
        only honest way to expose a limit whose input requires camera calibration
        the vehicle may not have.
        """
        offset = ctx.lateral_offset_m
        if offset is None or not math.isfinite(offset):
            self.lane_offset_available = False
            return
        self.lane_offset_available = True
        if abs(offset) > self.limits.max_lateral_offset_m:
            mitigated.append(
                "lane_departure_%.2fm_above_%.2fm" % (abs(offset), self.limits.max_lateral_offset_m)
            )

    def _steering_ceiling_rad(self, ego_speed_mps: float) -> float:
        """Largest road-wheel angle whose steady-state lateral accel is in budget."""
        lim = self.limits
        ceiling = lim.max_steering_angle_rad
        if ego_speed_mps > 1.0:
            ceiling = min(
                ceiling,
                math.atan(
                    lim.max_lateral_accel_mps2 * lim.wheelbase_m / (ego_speed_mps * ego_speed_mps)
                ),
            )
        return ceiling

    def _limit_steering(
        self,
        steering: float,
        ego_speed_mps: float,
        dt_s: float,
        mitigated: List[str],
        advisory: List[str],
    ) -> float:
        """Apply the lateral-acceleration cap and the road-wheel slew-rate limit.

        Both are enforced on the PHYSICAL angle, ``steering * max_road_wheel_rad``,
        not on the normalised number.

        The ``steering_rate`` finding is raised against the PREVIOUS REQUEST, not
        against the arbiter's own previous output.  Comparing to the output made the
        arbiter blame the planner for the arbiter's own hold: in a minimum-risk
        manoeuvre it replaced the steering with a held angle and then reported
        ``steering_rate_1.12_above_0.50`` every frame, a self-sustaining fault that
        fed the DISENGAGE counter.
        """
        lim = self.limits
        requested_rad = _clamp(steering, -1.0, 1.0) * lim.max_road_wheel_rad
        # While the arbiter is HOLDING the steering itself (MRM/DISENGAGE) the
        # planner's request is not actuated at all, so a finding about it is a
        # report, not a reason to stay in the manoeuvre.
        sink = advisory if self._lat_override_active else mitigated

        max_step = lim.max_steering_rate_rad_s * dt_s
        request_delta = requested_rad - self._prev_request_rad
        if abs(request_delta) > max_step + 1e-9:
            sink.append(
                "steering_rate_%.2f_above_%.2f"
                % (abs(request_delta) / dt_s, lim.max_steering_rate_rad_s)
            )
        self._prev_request_rad = requested_rad

        angle_rad = requested_rad
        ceiling = self._steering_ceiling_rad(ego_speed_mps)
        if abs(angle_rad) > ceiling + 1e-9:
            lateral = (ego_speed_mps ** 2) * math.tan(abs(angle_rad)) / lim.wheelbase_m
            sink.append(
                "lateral_accel_%.1f_above_%.1f" % (lateral, lim.max_lateral_accel_mps2)
            )
            angle_rad = _clamp(angle_rad, -ceiling, ceiling)

        angle_rad = self._slew(angle_rad, dt_s)
        return _clamp(angle_rad / lim.max_road_wheel_rad, -1.0, 1.0)

    def _slew(self, target_rad: float, dt_s: float) -> float:
        """Rate-limit a road-wheel angle against the arbiter's own last output."""
        max_step = self.limits.max_steering_rate_rad_s * dt_s
        delta = target_rad - self._prev_steering_rad
        if abs(delta) > max_step + 1e-9:
            self.last_shaping.append("steering_slew_limited")
            return self._prev_steering_rad + math.copysign(max_step, delta)
        return target_rad

    # -------------------------------------------------------------- decision

    def _decide_state(
        self,
        faults: List[str],
        mitigated: List[str],
        hazard: bool,
        cmd_ok: bool,
        ego_ok: bool,
        aeb: bool,
    ) -> SafetyState:
        """Map this frame's findings onto a requested state, before latching.

        Only HEALTH FAULTS accumulate toward DISENGAGE.  A hazard is a traffic
        situation the arbiter is designed to handle, and a MITIGATED finding is a
        clamp that worked -- neither is evidence that the automation can no longer
        drive.

        The hazard input is the BOOLEAN, not the presence of hazard strings.  A
        short headway at a matched speed is a real and reportable fact about the
        traffic and it is printed in ``violations`` every frame it holds, but a
        vehicle following at its own policy gap has not degraded: making the
        string force LIMITED left the system permanently degraded behind any lead
        closer than its policy spacing, which is most leads, and a degraded state
        that never clears is the self-sustaining latch this module has produced
        three times.
        """
        lim = self.limits
        requested = SafetyState.NOMINAL
        if faults or mitigated:
            requested = SafetyState.LIMITED
        if hazard:
            requested = _worse(requested, SafetyState.LIMITED)
        if self._dropout_streak >= lim.limited_after_dropouts:
            requested = _worse(requested, SafetyState.LIMITED)
        if self._dropout_streak >= lim.mrm_after_dropouts:
            requested = _worse(requested, SafetyState.MIN_RISK_MANEUVER)
        if aeb:
            requested = _worse(requested, SafetyState.MIN_RISK_MANEUVER)
        if not cmd_ok:
            requested = _worse(requested, SafetyState.MIN_RISK_MANEUVER)
        if not ego_ok:
            requested = _worse(requested, SafetyState.MIN_RISK_MANEUVER)

        if requested is SafetyState.NOMINAL:
            self._clean_streak += 1
        else:
            self._clean_streak = 0

        # DISENGAGE hands a moving vehicle back to nobody, and there is no driver
        # in this loop.  A PERCEPTION loss -- however prolonged -- is answered by
        # a controlled stop, not by a hand-back, so it is reported as a fault
        # (it is one) but it is excluded from the counter that latches the
        # terminal state.  What remains able to disengage is the class of fault
        # that means the automation cannot be COMMANDED or cannot tell the time:
        # a missing ego state, a non-finite command, a missing plan, a clock that
        # went backwards.  Those are not survivable by braking.
        terminal = [f for f in faults if not f.startswith("perception_dropout")]
        if terminal:
            self._health_fault_streak += 1
            if self._health_fault_streak >= lim.disengage_after_frames:
                requested = SafetyState.DISENGAGE
        else:
            self._health_fault_streak = 0
        return requested

    @property
    def fault_streak(self) -> int:
        """Consecutive frames carrying an unrecovered HEALTH fault.

        This, and only this, is what ``disengage_after_frames`` counts.
        """
        return self._health_fault_streak

    def _latch(self, requested: SafetyState, manoeuvring: bool = True) -> SafetyState:
        """Escalate immediately; de-escalate once the manoeuvre is genuinely over.

        The hysteresis exists so that a condition flickering across a threshold
        does not flicker the state with it, and ``recovery_frames`` is the
        default wait.  But a state is a claim about what the system is DOING, and
        holding ``MIN_RISK_MANEUVER`` for half a second after the brake has been
        released is a false claim: the harness calls it
        ``unwarranted_authority_state`` and it is right to -- a system that says
        it has given up on the driving task while doing nothing of the sort will
        either hand the vehicle back for nothing or start braking for nothing.

        So the wait is skipped on the one condition that makes it meaningless:
        the frame is clean AND the arbiter is not braking.  There is then nothing
        for the state to be describing.
        """
        if _SEVERITY[requested] >= _SEVERITY[self._latched_state]:
            self._latched_state = requested
            return self._latched_state
        if self._latched_state is SafetyState.DISENGAGE:
            return self._latched_state
        # A MINIMUM-RISK MANOEUVRE IS HELD ONLY WHILE ONE IS BEING EXECUTED.
        # The state is a claim about what the system is doing, and the claim
        # "I have given up on the driving task" is false the moment the brake is
        # no longer at emergency grade and no emergency test is tripped.  The
        # ordinary hysteresis still applies to LIMITED, where the claim is only
        # "I am constraining the command" and a flicker costs nothing.
        if self._latched_state is SafetyState.MIN_RISK_MANEUVER and not manoeuvring:
            self._latched_state = requested
            return self._latched_state
        if self._clean_streak >= self.limits.recovery_frames:
            self._latched_state = requested
        return self._latched_state

    # --------------------------------------------------------------- output

    def _required_decel(
        self,
        state: SafetyState,
        aeb: bool,
        hazard: bool,
        lead: LeadAssessment | None,
        perception_ok: bool = True,
        ego_ok: bool = True,
        cmd_ok: bool = True,
        plan_missing: bool = False,
    ) -> float:
        """The deceleration the arbiter itself demands this frame, m/s^2 (>= 0).

        This is the arbiter's DEMAND, before jerk shaping and before it is
        combined with the incoming command.  Three sources, and no fourth:

        1. **A measured hazard.**  ``lead.required_decel_mps2``, which is already
           zero unless a closure has been measured and its lower confidence bound
           is clear of zero.  It is used as computed -- there is no floor forcing
           it up to a fixed emergency figure, because a floor is exactly how a
           40 m approach to a stopped car that needed 2.7 m/s^2 came to be braked
           at 8.0.
        2. **Blindness.**  Once the vehicle has been unable to see for
           ``blind_hold_frames`` it makes a controlled stop at ``mrm_decel_mps2``.
           Before that the last avoidance demand is HELD: a detector that produced
           nothing has not produced evidence of an empty road.
        3. **A command or an ego state the arbiter cannot use.**  There is then no
           basis for any driving decision at all, so the vehicle stops, again at
           the controlled rate.

        A detection miss with HEALTHY perception is the second case's civilian
        cousin and is handled the same way, for ``miss_hold_frames``.
        """
        lim = self.limits
        required = 0.0
        if lead is not None and lead.required_decel_mps2 > 0.0:
            required = min(
                lim.max_deceleration_mps2,
                max(lim.aeb_min_decel_mps2, lead.required_decel_mps2 * lim.aeb_decel_margin)
                if aeb
                else lead.required_decel_mps2,
            )
            self._held_avoid_mps2 = required
            self._miss_frames = 0
        elif lead is None or not perception_ok:
            # Nothing measurable this frame.  Hold what the last measurement
            # justified, for as long as a hold is defensible, then let it go.
            self._miss_frames += 1
            window = lim.blind_hold_frames if not perception_ok else lim.miss_hold_frames
            if self._miss_frames > window:
                self._held_avoid_mps2 = 0.0
            required = self._held_avoid_mps2
        else:
            # An in-path object is being measured and it requires nothing.
            self._held_avoid_mps2 = 0.0
            self._miss_frames = 0

        if not perception_ok and self._blind_frames > lim.blind_hold_frames:
            required = max(required, lim.mrm_decel_mps2)
        if not ego_ok or not cmd_ok:
            required = max(required, lim.mrm_decel_mps2)
        if plan_missing:
            # NO PLAN MEANS NOTHING IS DRIVING.  The absence of a plan is not the
            # absence of a constraint: the planner threw, or the loop has ended
            # and this is the last command that will ever be written.  There is
            # no hold period for it -- a hold waits for evidence that might
            # arrive, and no evidence is coming -- so the controlled stop begins
            # on the frame the plan goes missing.  Whatever the arbiter was
            # already demanding is kept, because a stop that starts by releasing
            # the brake is not a stop.
            required = max(required, lim.mrm_decel_mps2)
        if state is SafetyState.DISENGAGE:
            required = max(required, lim.mrm_decel_mps2)
        return min(required, lim.max_deceleration_mps2)

    def _synthesise_command(
        self,
        cmd_in: ControlCommand,
        steering: float,
        state: SafetyState,
        aeb: bool,
        hazard: bool,
        lead: LeadAssessment | None,
        dt_s: float,
        cmd_ok: bool,
        ego_ok: bool,
        ego_speed_mps: float,
        perception_ok: bool,
        mitigated: List[str],
        plan_missing: bool = False,
    ) -> ControlCommand:
        """Build the command that actually reaches the actuators.

        The invariant, enforced rather than asserted: the output is never MORE
        energetic than the input.  ``throttle_out <= throttle_in`` always, and
        ``brake_out >= brake_in`` **except** on a veto frame -- see the module
        docstring.  The arbiter's own demand is jerk shaped here, at the same
        ceilings the primary path uses, so the arbiter can never be the source of
        a head-toss the occupant cannot brace for.

        Steering in MRM/DISENGAGE is HELD, not zeroed: see :meth:`_mrm_steering`.
        """
        lim = self.limits
        throttle = cmd_in.throttle
        brake = cmd_in.brake

        if state is not SafetyState.NOMINAL:
            throttle = min(throttle, lim.limited_throttle_cap)

        target = self._required_decel(
            state, aeb, hazard, lead, perception_ok, ego_ok, cmd_ok, plan_missing
        )
        demand = self._shaper.step(target, dt_s)
        if demand > 0.0:
            throttle = 0.0
            brake = max(brake, _clamp(demand / lim.brake_authority_mps2, 0.0, 1.0))

        if not cmd_ok:
            # A corrupt command tells us nothing about intent; fall back entirely
            # to the arbiter's own controlled stop.
            throttle = 0.0
            brake = max(brake, _clamp(lim.mrm_decel_mps2 / lim.brake_authority_mps2, 0.0, 1.0))

        if _SEVERITY[state] >= _SEVERITY[SafetyState.MIN_RISK_MANEUVER]:
            steering = self._mrm_steering(ego_speed_mps, dt_s)

        # Output shaping. Recorded but not escalating:
        #   * the throttle apply-rate limit -- that direction removes energy;
        #   * the brake RELEASE-rate floor on an INHERITED brake -- that direction
        #     only holds the brake on for longer.
        # There is deliberately no brake apply-rate limit here; the jerk shaper
        # above owns the rate at which the arbiter's own demand is built.
        throttle_cap = self._prev_throttle + lim.throttle_rate_per_s * dt_s
        if throttle > throttle_cap:
            self.last_shaping.append("throttle_rate_limited")
            throttle = throttle_cap
        brake_floor = self._prev_brake - lim.brake_release_rate_per_s * dt_s
        if not perception_ok:
            # NO EVIDENCE, NO RELEASE.  While the vehicle cannot see, nothing it
            # can measure supports the claim that the hazard it was braking for
            # has gone, so the brake is held at what it was.  This is also the
            # exit contract (ADAS-DEC-21): leaving the loop, a SIGTERM and a
            # source loss are the same event from the actuators' point of view,
            # the arbiter is called one last time with a failed perception
            # status, and whatever it returns is what stays latched on the
            # actuators.  Returning even a fraction less brake than the previous
            # frame would be releasing a brake mid-intervention on the way out.
            brake_floor = max(brake_floor, self._prev_brake)
        if brake < brake_floor:
            self.last_shaping.append("brake_release_limited")
            brake = brake_floor

        brake = max(brake, _clamp(cmd_in.brake, 0.0, 1.0))

        # ------------------------------------------------------------ the veto
        # The one place the arbiter is allowed to make the output LESS energetic
        # in the braking direction.  Perception is healthy, the ego state and the
        # command are usable, the vehicle is moving, PERCEPTION REPORTED NO OBJECT
        # ANYWHERE, and the arbiter is demanding nothing of its own -- which,
        # because of the hold above, also means it has not been braking for
        # something that merely dropped out of the track list.  A brake arriving
        # from upstream then rests on no measurement any part of this system can
        # make.
        #
        # The condition is "no track at all", not "no track in my corridor", and
        # the difference is the whole safety of the veto.  The arbiter's in-path
        # geometry is its own, and the point of it being its own is that it may
        # disagree with the planner's; if it vetoed on that disagreement it would
        # be discarding a brake the primary path applied for an object it can see
        # perfectly well and merely judges to be out of the way.  An empty track
        # list is not a judgement, it is the absence of evidence in both stacks at
        # once.
        #
        # It is a hard clamp, not a ramp.  Refusing authority is instantaneous:
        # there is nothing to ramp down from, because the deceleration should
        # never have been commanded.  Releasing a brake is not a comfort hazard.
        if (
            perception_ok
            and ego_ok
            and cmd_ok
            and not self._tracks_seen
            and demand <= 0.0
            and ego_speed_mps > lim.kinematics_min_speed_mps
        ):
            cap = _clamp(lim.no_hazard_decel_cap_mps2 / lim.brake_authority_mps2, 0.0, 1.0)
            if brake > cap:
                vetoed = (brake - cap) * lim.brake_authority_mps2
                if vetoed >= lim.comfort_decel_mps2:
                    mitigated.append(
                        "brake_%.2f_vetoed_empty_road" % (brake * lim.brake_authority_mps2)
                    )
                else:
                    self.last_shaping.append("brake_vetoed_no_hazard")
                brake = cap

        throttle = _clamp(throttle, 0.0, 1.0)
        brake = _clamp(brake, 0.0, 1.0)
        if brake > 0.0:
            throttle = 0.0
        return ControlCommand(
            throttle=throttle, brake=brake, steering=_clamp(steering, -1.0, 1.0)
        )

    def _mrm_steering(self, ego_speed_mps: float, dt_s: float) -> float:
        """Steering for a minimum-risk manoeuvre: HOLD the path, do not straighten.

        An MRM brakes to a stop on the CURRENT path.  Replacing the steering with a
        latch that was only ever written in NOMINAL meant that any MRM entered in a
        bend -- and a bend is where the lateral clamps are active, so NOMINAL had
        not been reached for many frames -- commanded ``steering = 0.0`` while
        braking.  It steered straight into the corner and then reported
        ``lateral_accel_8.4_above_4.5`` against its own output.

        What it does instead: hold the last ACTUATED angle (recorded every frame,
        in every state), re-check it against the lateral-acceleration ceiling at the
        CURRENT speed -- the ceiling relaxes as the vehicle slows -- and slew toward
        straight only once the vehicle is essentially stopped.
        """
        lim = self.limits
        held_rad = self._last_commanded_steering * lim.max_road_wheel_rad
        if ego_speed_mps < lim.mrm_straighten_speed_mps:
            held_rad = 0.0
        else:
            ceiling = self._steering_ceiling_rad(ego_speed_mps)
            held_rad = _clamp(held_rad, -ceiling, ceiling)
        return _clamp(self._slew(held_rad, dt_s) / lim.max_road_wheel_rad, -1.0, 1.0)

    def _describe(
        self,
        state: SafetyState,
        violations: List[str],
        lead: LeadAssessment | None,
        aeb: bool,
    ) -> str:
        """One-line human-readable summary; also what goes into the logs."""
        parts = [state.value]
        if aeb:
            parts.append("AEB")
        if lead is not None:
            parts.append(
                "lead#%d d=%.1fm rate=%s ttc=%s a_req=%.1f src=%s"
                % (
                    lead.track_id,
                    lead.distance_m,
                    (
                        "%+.1fm/s(lcb%+.1f)" % (lead.range_rate_mps, lead.closing_lcb_mps)
                        if lead.rate_is_measured
                        else "unmeasured(%d/%d)"
                        % (lead.rate_updates, self.limits.evidence.min_samples)
                    ),
                    "inf" if math.isinf(lead.ttc_s) else "%.2fs" % lead.ttc_s,
                    lead.required_decel_mps2 if math.isfinite(lead.required_decel_mps2) else -1.0,
                    lead.source.value,
                )
            )
            if lead.coasting:
                parts.append("coasted=%d" % lead.coast_frames)
        else:
            parts.append("no_in_path_lead")
        if violations:
            parts.append("violations=" + ",".join(violations[:6]))
        if not self.lane_offset_available:
            parts.append("lane_offset_unavailable")
        if self.last_shaping:
            parts.append("shaped=" + ",".join(sorted(set(self.last_shaping))))
        return " | ".join(parts)
