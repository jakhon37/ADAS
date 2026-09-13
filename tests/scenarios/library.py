"""The scenario library: the executable form of the safety specification.

Every entry states, in its ``physics`` field, the arithmetic that makes its
expected outcome the correct one.  Nothing here was derived by running the
current arbiter and writing down what it did; the numbers come from stopping
distances, closing rates and the authority limits declared in
:mod:`tests.scenarios.plant`.

Scenarios sit ON the failure boundary
-------------------------------------
The first version of this library was rejected by its own backtest: the four
cases that named "round-2 missed braking" as their guard all PASSED on the
commit that collides, because they handed the ego a free second of zero closing
rate before the lead braked, and the stationary family sat 16-32 m clear of the
range at which contact actually occurs.  A scenario 20 m clear of the boundary
proves nothing: it cannot tell a system that is 1 m from failing from one that
is 20 m from failing, and it cannot detect a regression until the regression is
already catastrophic.

So every family below is placed by MEASUREMENT.  The boundaries were located by
sweeping :mod:`tests.scenarios.sweep` and the closed-loop scenario runner
against the two committed arbiter versions with known failures, ``25e3ba5``
(phantom braking) and ``1ce4886`` (missed braking), and each family now carries
at least one case clearly INSIDE the measured failing region and one just
OUTSIDE it.  The measured numbers are quoted in the ``physics`` strings and
collected in ``docs/SAFETY_SPEC.md``.

The three boundaries that place this library
--------------------------------------------
=================================  =========================================
measurement                        boundary
=================================  =========================================
Full-authority stop through the    10 m/s -> 6.67 m, 15 m/s -> 14.70 m,
plant, 8 m/s^2 with the 0.15 s     20 m/s -> 25.85 m, 25 m/s -> 40.13 m.
brake rise.                        A stationary obstacle closer than this
                                   cannot be avoided by ANY system, so a
                                   scenario placed there is mis-specified.
Constant-range phantom, 25e3ba5.   Emergency authority (>= 3.5 m/s^2 or
                                   MIN_RISK_MANEUVER) for every gap <= 43 m
                                   at 20 m/s, <= 26 m at 15 m/s, <= 66 m at
                                   25 m/s; nothing at 44 m and beyond.
Lead braking at 6 m/s^2 from a     Avoidable CONTACT for d0 = 16-26 m at
matched-speed follow, 1ce4886.     20 m/s; clears by 0.28 m at 27 m and
                                   2.22 m at 32 m.  25e3ba5 never contacts
                                   here but clears by only 0.31-0.40 m over
                                   16-20 m.
=================================  =========================================

Both error directions, in every family
--------------------------------------
Round 1 of the fixes satisfied only the "did not brake for nothing" half of the
specification and round 2 only the "did brake in time" half, which is how the
arbiter came to oscillate.  Each entry below therefore records, in its
``physics``, what a TOO-AGGRESSIVE system does to it and what a TOO-PASSIVE one
does.  Where one direction is structurally impossible -- at the avoidability
boundary full authority is the only defensible command, so there is no
aggressive failure to detect -- the entry says so and names the sibling case in
the same family that carries that direction.

Coverage: which historical failure each family guards
-----------------------------------------------------
``round-1 phantom AEB``
    the six :func:`_constant_range` cases at 12, 20, 30, 40, 52 and 70 m,
    :func:`reid_during_steady_follow`, :func:`range_jump_during_steady_follow`,
    ``empty_road_nominal``
``round-1 disengagement on an empty road``
    ``empty_road_nominal``, ``empty_road_dropout``
``round-1 lane-error blindness``
    ``lane_error_with_closing_lead``
``round-2 missed braking against a lead braking at 6 m/s^2``
    the six :func:`_lead_brakes_hard` cases
``missed braking against a stopped obstacle``
    the six :func:`_stationary` cases
``the unpoliced sub-emergency band``
    every case carrying ``max_commanded_decel_mps2 = 3.0``: a deceleration
    between the 3.0 m/s^2 comfort limit and the 3.5 m/s^2 emergency threshold
    is invisible to the phantom test, so the constant-range family pins it with
    an explicit ceiling at comfort instead
``self-sustaining loops (steering_rate -> plan_accel -> lane_departure)``
    every scenario carrying ``recover_within_frames``
``ADAS-DEC-21 latched command at exit``
    ``terminate_during_emergency``, ``source_loss_mid_run``
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from adas.core.models import SafetyState

from tests.scenarios.plant import (
    DT_S,
    LANE_WIDTH_M,
    LeadSpec,
    PerceptionSpec,
    PlantConfig,
    adjacent_lane_object,
    dt_with_overruns,
    lead_accelerates,
    lead_brake_then_release,
    lead_brakes,
    lead_constant_speed,
    lead_stationary,
    noisy_perception,
)
from tests.scenarios.plant import constant_bend
from tests.scenarios.scenario import DEFAULT_STACK, Expectation, Scenario

# --------------------------------------------------------------------------- #
# Shared arithmetic, quoted in the physics strings below
# --------------------------------------------------------------------------- #

STOP_DISTANCE_M = {10.0: 6.67, 15.0: 14.70, 20.0: 25.85, 25.0: 40.13}
"""Distance to standstill under full authority THROUGH THE PLANT, metres.

Measured by driving :class:`tests.scenarios.plant.Plant` at ``brake = 1.0`` from
each speed, so it includes the 0.15 s brake rise; the idealised ``v^2 / 2a``
figures are 6.25 / 14.06 / 25.00 / 39.06 m.

This is NOT the avoidability boundary, and treating it as one is how the
stationary family came to be mis-placed.  It is the distance covered by a brake
that arrives instantaneously at full pedal, on the frame the obstacle first
exists, with no measurement in between.  No compliant system can do any of those
three things.  Placement uses :data:`AVOIDABILITY_M` instead; this table stays
because the physics strings quote both, and the gap between them IS the
specification's own honesty margin.
"""

AVOIDABILITY_M = {10.0: 9.37, 15.0: 18.77, 20.0: 31.30, 25.0: 46.95}
"""Distance inside which a stationary obstacle cannot be avoided BY ANY CORRECT
SYSTEM, metres.  This is what the stationary family is placed against.

Measured with :func:`tests.scenarios.scenario.feasibility`, which charges the
two costs :data:`STOP_DISTANCE_M` ignores and which no system can avoid paying:

* the brake is built at :data:`tests.scenarios.oracle.EMERGENCY_JERK_MPS3`
  rather than stepped to full in one frame.  A step is 160 m/s^3 and this
  specification's own ``excess_jerk`` finding forbids it, so an oracle that
  judged avoidability against a step would place every scenario on a boundary
  no compliant system is allowed to reach.  At 20 m/s the ramp costs 3.45 m.
* the decision cannot be taken before the second DISTINCT capture of the
  obstacle -- frames 0 and 1 both read capture 0 at the measured 55 ms sense
  latency, so frame 2 is the first honest one.  That is 0.10 s, and 2.00 m at
  20 m/s.  A system that acts sooner is acting on the stationary prior, which is
  exactly the constant-range phantom this harness exists to catch: a scenario
  passable only by braking on that prior is not testing competence, it is
  rewarding the defect.

Zero-latency (from the hazard frame, jerk-limited) is 8.37 / 17.27 / 29.30 /
44.45 m; the table above adds the two-frame measurement floor.  A 2.00 m
clearance is therefore first reachable at **21.0 m at 15 m/s, 33.5 m at 20 m/s
and 49.0 m at 25 m/s**, and every stationary case below sits outside those.

RE-MEASURED in this revision, and it moved: the previous placement used
:data:`STOP_DISTANCE_M` and put five cases INSIDE the real boundary --
``stationary_15mps_at_15m`` (best reachable -0.21 m against a 0.25 m
requirement), ``stationary_20mps_at_26m`` (-0.30 against 0.10),
``stationary_20mps_at_29m`` (-0.23 against 2.00), ``stationary_25mps_at_42m``
(-0.17 against 1.00) and ``lane_error_with_closing_lead`` (-0.01 against 2.00).
Every one of them was unsatisfiable, and an unsatisfiable expectation is as
useless as an absent one, because the only way to pass it is to brake before the
measurement exists.
"""

POLICY_GAP_20MPS_M = 52.0
"""The vehicle's own following policy at 20 m/s: 12 m + 2.0 s x 20 m/s.

A gap at or beyond this gives the longitudinal law nothing to correct, so a
constant-range scenario at 52 m or 70 m must produce a command of exactly zero.
Inside it, opening the gap is correct behaviour and is bounded by comfort.
"""

COMFORT_DECEL_MPS2 = 3.0
"""Deceleration above which braking stops being headway keeping.

Re-stated from :data:`tests.scenarios.oracle.COMFORT_DECEL_MPS2` so the ceilings
below can be read without opening the oracle.  It is 0.5 m/s^2 BELOW the
emergency threshold the phantom test uses, and that half-metre band is the one
the backtest found unpoliced: a system can sit at 3.4 m/s^2 for a whole run
against a lead that never moved and no assertion in this harness notices.  The
constant-range family closes it with an explicit ceiling at comfort.
"""

REQUIRED_CLEARANCE_M = 2.0
"""Clearance a correct intervention preserves, metres.

:data:`tests.scenarios.oracle.REQUIRED_CLEARANCE_M`.  It is only ever demanded
where the plant can actually deliver it -- ``d0 - STOP_DISTANCE_M >= 2.0`` for a
stationary obstacle -- so no scenario asks for a margin the vehicle does not
have.
"""

JUSTIFICATION_WINDOW_FRAMES = 10
"""The oracle's justification look-back, frames (0.5 s at 20 Hz).

Used here as the allowance for over-strong braking at the ONSET of a lead's
deceleration.  When a matched-speed lead starts braking at 6 m/s^2 the true
requirement climbs from 0 to about 4.3 m/s^2 within one frame and to 5.4 m/s^2
within half a second, taking the justified ceiling from 3.5 to roughly 8.5
m/s^2 with it.  Full authority is therefore above the instantaneous ceiling for
at most that half second and fully justified after it, so ten frames of
over-strong response is a reaction transient and an eleventh is a policy.
"""


def _gap_opening_speed_floor_mps(ego_mps: float, gap_m: float) -> float:
    """Lowest speed a comfort-limited gap-opening manoeuvre can need, m/s.

    To grow a gap by ``D`` metres against a lead holding station, the ego must
    fall behind by ``D`` metres of travel and then catch back up.  Decelerating
    at the 3.0 m/s^2 comfort limit to a speed deficit ``dv`` and recovering at
    the plant's 2.5 m/s^2 costs ``dv^2 / 6 + dv^2 / 5`` metres, so

        dv = sqrt(D / (1/6 + 1/5))

    is the largest deficit any comfort-limited manoeuvre needs.  Anything below
    ``ego - dv`` is the system braking for something the world is not doing, and
    is reported in metres per second rather than in state labels -- which is the
    only unit in which "it braked for nothing" is arguable.

    Args:
        ego_mps: Cruise speed the ego holds before the correction.
        gap_m: Gap at the start of the run, metres.

    Returns:
        The speed floor, m/s, never below zero.
    """
    to_open = max(0.0, POLICY_GAP_20MPS_M * (ego_mps / 20.0) - gap_m)
    dv = (to_open / (1.0 / 6.0 + 1.0 / 5.0)) ** 0.5
    return max(0.0, ego_mps - dv)


# --------------------------------------------------------------------------- #
# Family 1: constant range -- the round-1 phantom boundary
# --------------------------------------------------------------------------- #


def _constant_range(gap_m: float, inside_phantom_region: bool) -> Scenario:
    """A matched-speed follow at a fixed gap: the phantom-braking boundary case.

    The lead holds the ego's speed for the whole run, so the true closing rate
    is identically zero and the true collision-avoidance requirement is
    identically zero at every frame.  Nothing the ego can do changes that, and
    nothing in the scene can become contact without the lead first acting.

    Args:
        gap_m: The constant true gap, metres.
        inside_phantom_region: Whether the measured 25e3ba5 phantom region
            covers this gap.  Only changes the wording; the requirement is
            derived from the kinematics either way.

    Returns:
        The scenario.
    """
    at_or_beyond_policy = gap_m >= POLICY_GAP_20MPS_M
    floor = 19.0 if at_or_beyond_policy else _gap_opening_speed_floor_mps(20.0, gap_m)
    ceiling = 0.5 if at_or_beyond_policy else COMFORT_DECEL_MPS2
    return Scenario(
        name="constant_range_%02dm" % gap_m,
        summary="15 s of matched-speed following at 20 m/s and a constant %.0f m" % gap_m,
        guards="round-1 phantom AEB",
        physics=(
            "Both vehicles hold 20 m/s for 15 s. The true closing rate is identically zero and "
            "the true gap is identically %.0f m, so the collision-avoidance requirement is "
            "identically zero on every one of the 300 frames and no measurement anywhere in "
            "this run justifies emergency authority. %s "
            "MEASURED BOUNDARY: the phantom-braking commit 25e3ba5 applies emergency authority "
            "(>= 3.5 m/s^2 or MIN_RISK_MANEUVER) at every constant-range gap up to 43 m at "
            "20 m/s and none at 44 m or beyond, dragging the ego from 20 m/s down to 13.80, "
            "14.83, 15.86 and 17.27 m/s at 12, 20, 30 and 40 m; the missed-braking commit "
            "1ce4886 applies it up to 20 m and then holds 3.0-3.3 m/s^2 -- inside the "
            "unpoliced sub-emergency band -- all the way out to 45 m. This case sits %s that "
            "boundary. "
            "TOO AGGRESSIVE: peak commanded deceleration above %.1f m/s^2, a safety state "
            "worse than %s, or a speed floor below %.1f m/s, all of which are braking for a "
            "vehicle that has done nothing for 15 s. TOO PASSIVE: never letting go -- a final "
            "speed below %.1f m/s or a degraded state that outlives the (non-existent) "
            "disturbance, which is the self-sustaining loop that produced round 1."
            % (
                gap_m,
                (
                    "The gap is at or beyond the vehicle's own %.0f m policy spacing "
                    "(12 m + 2.0 s x 20 m/s), so the headway requirement is zero too and the "
                    "only correct brake command is exactly zero."
                    % POLICY_GAP_20MPS_M
                    if at_or_beyond_policy
                    else
                    "The gap is inside the vehicle's own %.0f m policy spacing "
                    "(12 m + 2.0 s x 20 m/s), so opening it is correct behaviour -- but "
                    "opening it is headway keeping, which is bounded by the 3.0 m/s^2 comfort "
                    "limit, and opening %.0f m at comfort and recovering at the plant's "
                    "2.5 m/s^2 costs at most %.2f m/s of speed."
                    % (POLICY_GAP_20MPS_M, POLICY_GAP_20MPS_M - gap_m, 20.0 - floor)
                ),
                "INSIDE" if inside_phantom_region else "OUTSIDE",
                ceiling,
                "NOMINAL" if at_or_beyond_policy else "LIMITED",
                floor,
                19.0 if at_or_beyond_policy else 18.0,
            )
        ),
        frames=300,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=20.0,
            accel_fn=lead_constant_speed(),
            label="lead keeping pace at %.0f m" % gap_m,
        ),
        expect=Expectation(
            no_collision=True,
            forbid_emergency_intervention=True,
            max_commanded_decel_mps2=ceiling,
            max_safety_state=(
                SafetyState.NOMINAL if at_or_beyond_policy else SafetyState.LIMITED
            ),
            min_speed_floor_mps=round(floor, 2),
            min_final_speed_mps=19.0 if at_or_beyond_policy else 18.0,
            recover_within_frames=None if at_or_beyond_policy else 120,
        ),
    )


# --------------------------------------------------------------------------- #
# Family 2: a lead braking hard -- the round-2 missed-braking boundary
# --------------------------------------------------------------------------- #


def _lead_brakes_hard(ego_mps: float, gap_m: float, note: str) -> Scenario:
    """A matched-speed follow in which the lead brakes at 6 m/s^2 from FRAME 0.

    The first version of this family started the lead's brake at t = 1.0 s.
    That second of zero closing rate is a free second: it lets the ego's own
    headway law settle before anything happens, and it lifted the whole family
    clear of the region in which the round-2 arbiter collides.  The lead
    therefore brakes from the first frame here, which is also the harder and
    more honest case -- a real lead does not announce itself.

    Args:
        ego_mps: Ego and initial lead speed, m/s.
        gap_m: Initial gap, metres.
        note: Where this case sits relative to the measured contact region.

    Returns:
        The scenario.
    """
    lead_stop_m = ego_mps * ego_mps / 12.0
    budget_m = gap_m + lead_stop_m - REQUIRED_CLEARANCE_M
    ego_stop_m = STOP_DISTANCE_M[ego_mps]
    slack_s = (budget_m - ego_stop_m) / ego_mps
    return Scenario(
        name="lead_brakes_6mps2_ego%02d_at_%02dm" % (ego_mps, gap_m),
        summary="matched %.0f m/s follow at %.0f m; the lead brakes at 6 m/s^2 from frame 0"
        % (ego_mps, gap_m),
        guards="round-2 missed braking",
        physics=(
            "Both vehicles start at %.0f m/s, so the closing rate at frame 0 is exactly zero "
            "and the lead's 6 m/s^2 deceleration begins on the first frame -- there is no free "
            "second in which the ego's headway law can settle first. The lead stops after "
            "%.2f s having covered %.1f m, so the ego has %.0f + %.1f - %.1f = %.1f m in which "
            "to stop from %.0f m/s, against the %.2f m the plant needs at full authority "
            "including its 0.15 s brake rise. The intervention may therefore begin as late as "
            "%.2f s after the lead's brake light and still hold the %.1f m clearance. Because "
            "both vehicles start matched, the gap can only shrink while the ego decelerates "
            "less than the lead: an ego braking flat out from frame 0 keeps %.2f m, so the "
            "%.1f m clearance demanded here has %.1f m of headroom and is not a knife edge. "
            "MEASURED BOUNDARY: %s "
            "TOO PASSIVE: contact, or a minimum true gap under %.1f m. TOO AGGRESSIVE: the "
            "true requirement is 0 at frame 0 and only about %.2f m/s^2 one frame later, so "
            "the justified ceiling starts at %.2f m/s^2; full authority applied for more than "
            "the oracle's own %d-frame (0.5 s) justification window, before the lead has shed "
            "enough speed to warrant it, is over-braking and moves the collision to the "
            "vehicle behind."
            % (
                ego_mps,
                ego_mps / 6.0,
                lead_stop_m,
                gap_m,
                lead_stop_m,
                REQUIRED_CLEARANCE_M,
                budget_m,
                ego_mps,
                ego_stop_m,
                slack_s,
                REQUIRED_CLEARANCE_M,
                gap_m - 0.01,
                REQUIRED_CLEARANCE_M,
                gap_m - 0.01 - REQUIRED_CLEARANCE_M,
                note,
                REQUIRED_CLEARANCE_M,
                _required_one_frame_in(ego_mps, gap_m),
                max(3.5, _required_one_frame_in(ego_mps, gap_m) * 1.5 + 0.5),
                JUSTIFICATION_WINDOW_FRAMES,
            )
        ),
        frames=300,
        ego_speed_mps=ego_mps,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=ego_mps,
            accel_fn=lead_brakes(6.0, start_s=0.0),
            label="lead braking at 6 m/s^2 from frame 0",
        ),
        expect=Expectation(
            no_collision=True,
            min_clearance_m=REQUIRED_CLEARANCE_M,
            must_intervene=True,
            unjustified_brake_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
        ),
    )


def _required_one_frame_in(ego_mps: float, gap_m: float) -> float:
    """Deceleration the ego needs one frame after the lead starts braking.

    Closed form of the oracle's causal question at frame 1: the lead is at
    ``v - 6 dt`` still decelerating at 6 m/s^2, so it will cover
    ``(v - 6 dt)^2 / 12`` more metres, and the ego must stop inside the gap plus
    that distance less the 2 m clearance.  Quoted in the physics strings so that
    the "too aggressive" ceiling is a number rather than an adjective.
    """
    dt = 0.05
    lead_v = ego_mps - 6.0 * dt
    gap = gap_m - 0.5 * 6.0 * dt * dt
    budget = gap + lead_v * lead_v / 12.0 - REQUIRED_CLEARANCE_M
    return (ego_mps * ego_mps) / (2.0 * budget)


# --------------------------------------------------------------------------- #
# Family 3: a stopped obstacle -- the avoidability boundary
# --------------------------------------------------------------------------- #


def _stationary(
    ego_mps: float,
    gap_m: float,
    frames: int,
    clearance_m: float,
    note: str,
    over_brake_ceiling_mps2: Optional[float] = None,
    tail_frames_allowed: int = 0,
) -> Scenario:
    """A parked vehicle in the ego lane, visible from the first frame.

    Placed relative to :data:`STOP_DISTANCE_M`, because a stationary obstacle is
    the one case whose outcome the arithmetic can settle before the system gets
    a vote: at 20 m/s a car 25 m ahead is already unavoidable, so a "missed
    brake" reported there is a defect in the scenario, not in the system.

    Args:
        ego_mps: Ego speed, m/s.
        gap_m: Range to the parked car at frame 0, metres.
        frames: Episode length.
        clearance_m: Minimum true gap the run must preserve.  Never more than
            ``gap_m - STOP_DISTANCE_M[ego_mps]``, which is all the vehicle has.
        note: The measured behaviour of the two broken commits here.
        over_brake_ceiling_mps2: Peak commanded deceleration this case permits,
            or ``None`` where full authority is the only defensible answer.
        tail_frames_allowed: Frames on which the command may exceed the oracle's
            decaying justified ceiling.  Non-zero only where the requirement at
            frame 0 is already within a few per cent of full authority: the
            command then cannot be modulated below the ceiling on the way in,
            and the only frames that can exceed it are the last few of a
            completed stop, where the brake has done its job and the
            requirement has collapsed while a jerk-limited command is still
            being released.  Bounded by the oracle's own 0.5 s justification
            window, so a release transient is forgiven and a policy is not.

    Returns:
        The scenario.
    """
    stop_m = STOP_DISTANCE_M[ego_mps]
    boundary_m = AVOIDABILITY_M[ego_mps]
    best_m = gap_m - boundary_m
    needed = (ego_mps * ego_mps) / (2.0 * max(0.01, gap_m - REQUIRED_CLEARANCE_M))
    return Scenario(
        name="stationary_%02dmps_at_%02dm" % (ego_mps, gap_m),
        summary="stopped vehicle %.0f m ahead, ego at %.0f m/s on an empty straight"
        % (gap_m, ego_mps),
        guards="missed braking against a stopped obstacle",
        physics=(
            "The obstacle is stationary and dead ahead, so the closing rate is exactly the ego "
            "speed and never changes sign: there is no reading of the measurements under which "
            "doing nothing is correct. An idealised full-authority stop through the plant, "
            "8 m/s^2 behind a 0.15 s brake rise, covers %.2f m from %.0f m/s -- but no "
            "compliant system gets that: the brake must be built at the emergency jerk limit "
            "rather than stepped to full, and the decision cannot be taken before the second "
            "distinct capture of the obstacle, two frames in. Charging both puts the real "
            "avoidability boundary at %.2f m, so the best clearance ANY correct system can "
            "hold from %.0f m is %.2f m and this case requires %.1f m of it. A scenario placed "
            "inside %.2f m would be asking the system to brake before the measurement exists, "
            "which is the phantom this harness punishes everywhere else. A constant "
            "%.2f m/s^2 from frame 0 is enough to stop with the %.1f m standstill clearance "
            "intact. "
            "MEASURED: %s "
            "TOO PASSIVE: contact, or a minimum true gap below the %.1f m this case requires. "
            "%s"
            % (
                stop_m,
                ego_mps,
                boundary_m,
                gap_m,
                best_m,
                clearance_m,
                boundary_m,
                needed,
                REQUIRED_CLEARANCE_M,
                note,
                clearance_m,
                (
                    "TOO AGGRESSIVE: peak commanded deceleration above %.1f m/s^2. The true "
                    "requirement here is only %.2f m/s^2, below the %.1f m/s^2 comfort limit, "
                    "so this is not an emergency at all yet and full authority is a threefold "
                    "over-response -- the aggressive direction for the whole stationary family "
                    "is carried by this case."
                    % (over_brake_ceiling_mps2, needed, COMFORT_DECEL_MPS2)
                    if over_brake_ceiling_mps2 is not None
                    else
                    "TOO AGGRESSIVE: not detectable here, and deliberately so. At %.0f m the "
                    "requirement is %.2f m/s^2 or unbounded, so full authority is the only "
                    "defensible command and there is no over-response to find; the aggressive "
                    "direction for this family is carried by stationary_20mps_at_75m, where "
                    "the requirement is below comfort."
                    % (gap_m, min(8.0, needed))
                ),
            )
        ),
        frames=frames,
        ego_speed_mps=ego_mps,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=0.0,
            accel_fn=lead_stationary(),
            label="parked car at %.0f m" % gap_m,
        ),
        expect=Expectation(
            no_collision=True,
            min_clearance_m=clearance_m,
            must_intervene=True,
            max_commanded_decel_mps2=over_brake_ceiling_mps2,
            unjustified_brake_frames_allowed=tail_frames_allowed,
        ),
    )


# --------------------------------------------------------------------------- #
# Family 4: sensor fidelity -- noise, bias, and a frame period that is not 20 Hz
# --------------------------------------------------------------------------- #
#
# These exist because the plant grew the features and nothing used them.  A
# programmatic census of the previous revision of this file found
# ``range_noise_m > 0`` in 0 of 37 scenarios, ``range_bias_frac != 0`` in 0 of
# 37, and a frame period other than 50 ms in 0 of 37 -- and the census is now
# committed as :func:`coverage_census` and asserted in ``tests/test_scenarios``,
# so it cannot silently go back to zero.
#
# None of them is decoration:
#
# * the arbiter's corroboration window exists BECAUSE of a false positive
#   measured at +/-0.30 m of range noise, and a harness that cannot produce that
#   noise cannot certify the fix;
# * a mono camera's range is ``f*H/h``, so a wrong height prior biases every
#   range by the same fraction, in the same direction, for ever -- and no
#   corroboration window can see a bias, because a bias does not average out;
# * this board's own 200-frame run measured a median frame interval of 58.56 ms,
#   a p95 of 75.06 ms and a maximum of 174.33 ms, and the production log records
#   ``Frame 325 took 187.7 ms``.  A frame that takes 187 ms is 3.7 m of travel at
#   20 m/s during which the last command was still acting and nothing was
#   decided.

HISTORIC_RANGE_NOISE_M = 0.30
"""The range-noise figure the arbiter's corroboration logic was written against.

Not a round number chosen for looks: ``docs/SAFETY_SPEC.md`` and
:func:`tests.scenarios.plant.noisy_perception` both record that the historical
false positive was measured at +/-0.30 m.  Every noise scenario below uses it,
so a change to the corroboration logic is tested against the noise that provoked
it rather than against a number somebody liked.
"""

RATE_WINDOW_FRAMES = 5
"""Samples the plant's range-rate estimator fits over.

:attr:`tests.scenarios.plant.PerceptionSpec.rate_window_frames`.  Restated here
because the noise arithmetic below depends on it.
"""


def apparent_closure_from_noise_mps(
    noise_m: float = HISTORIC_RANGE_NOISE_M,
    window: int = RATE_WINDOW_FRAMES,
    dt_s: float = DT_S,
    sigmas: float = 3.0,
) -> float:
    """Largest spurious closing rate pure range noise can fabricate, m/s.

    A camera measures range; every closing rate downstream is a slope fitted to
    a range history, so range noise becomes rate noise amplified by the inverse
    of the fitting span.  For ``n`` equally spaced samples ``dt`` apart, the
    least-squares slope of a *constant* range has standard error

        SE = sigma / sqrt(Sxx),   Sxx = dt^2 * n (n^2 - 1) / 12

    which at ``sigma = 0.30 m``, ``n = 5``, ``dt = 0.05 s`` is
    ``0.30 / sqrt(0.025) = 1.90 m/s``.  Three of those is 5.69 m/s: a closure
    the scene does not contain, invented entirely by the sensor.

    This is the number the noise scenarios are placed against, and it is what
    makes them PASSABLE rather than a trap.  A spurious closure ``c`` at a true
    gap ``g`` demands ``c^2 / (2 (g - 2))`` to hold the 2 m clearance, so at the
    gaps used below the worst-case demand is

        20 m ->  5.69^2 / (2 x 18) = 0.90 m/s^2
        40 m ->  5.69^2 / (2 x 38) = 0.43 m/s^2

    -- both far under the 3.0 m/s^2 comfort limit.  A correct system therefore
    passes these cases even if it believes every noisy sample completely, and no
    correct system has to be clairvoyant to do so.

    Args:
        noise_m: Range-noise standard deviation, metres.
        window: Samples in the rate fit.
        dt_s: Sample spacing, seconds.
        sigmas: How many standard errors to report.

    Returns:
        The spurious closing rate, m/s.
    """
    n = max(2, int(window))
    sxx = dt_s * dt_s * n * (n * n - 1) / 12.0
    return float(sigmas) * float(noise_m) / math.sqrt(sxx)


def _decel_for_closure(closure_mps: float, gap_m: float) -> float:
    """Deceleration needed to hold :data:`REQUIRED_CLEARANCE_M` at a closure."""
    budget = max(0.01, gap_m - REQUIRED_CLEARANCE_M)
    return (closure_mps * closure_mps) / (2.0 * budget)


def _noisy_constant_range(gap_m: float, seed: int) -> Scenario:
    """A matched-speed follow at a fixed gap, seen through +/-0.30 m of noise.

    The world is exactly :func:`_constant_range`: the true closing rate is
    identically zero and the true requirement is identically zero on every
    frame.  The only difference is that the range the system is told is the true
    range plus a seeded, zero-mean Gaussian draw with the standard deviation
    that produced the historical false positive.

    This is the case the arbiter's corroboration window exists for, and until
    this scenario existed the harness could not produce it.

    Args:
        gap_m: The constant TRUE gap, metres.
        seed: Noise seed.  Fixed per scenario, so the run is reproducible; two
            gaps use different seeds so that a lucky draw cannot make the whole
            family pass.

    Returns:
        The scenario.
    """
    spurious = apparent_closure_from_noise_mps()
    worst = _decel_for_closure(spurious, gap_m)
    floor = _gap_opening_speed_floor_mps(20.0, gap_m)
    return Scenario(
        name="noisy_range_%02dm_030m_noise" % gap_m,
        summary="matched-speed follow at 20 m/s and %.0f m with +/-0.30 m of range noise"
        % gap_m,
        guards="round-1 phantom AEB (the +/-0.30 m range-noise false positive)",
        physics=(
            "Both vehicles hold 20 m/s for 15 s at a constant TRUE gap of %.0f m, so the true "
            "closing rate is identically zero and the true collision-avoidance requirement is "
            "identically zero on all 300 frames. What the system is told is that range plus a "
            "seeded zero-mean Gaussian of standard deviation %.2f m -- the figure at which the "
            "historical false positive was measured, and the reason the arbiter has a "
            "corroboration window at all. "
            "WHY THIS IS PASSABLE, which matters because an unsatisfiable expectation is as "
            "useless as an absent one: a least-squares slope over the estimator's %d-sample, "
            "%.2f s window has standard error sigma / sqrt(dt^2 n (n^2-1) / 12) = "
            "%.2f / sqrt(%.4f) = %.2f m/s, so three standard errors of pure noise is a "
            "spurious closure of %.2f m/s. Holding the %.1f m clearance against %.2f m/s of "
            "closure at %.0f m needs %.2f^2 / (2 x %.0f) = %.2f m/s^2, which is below the "
            "%.1f m/s^2 comfort limit. A system that believed every noisy sample completely "
            "would still not be entitled to emergency authority here, so this case is passable "
            "by any correct system and is not a demand for clairvoyance. "
            "TOO AGGRESSIVE: emergency authority, a state worse than LIMITED, a command above "
            "%.1f m/s^2, or a speed floor below %.1f m/s -- all of them braking for sensor "
            "noise. TOO PASSIVE: finishing below 18 m/s, i.e. a response to the noise that is "
            "never released."
            % (
                gap_m,
                HISTORIC_RANGE_NOISE_M,
                RATE_WINDOW_FRAMES,
                RATE_WINDOW_FRAMES * DT_S,
                HISTORIC_RANGE_NOISE_M,
                DT_S * DT_S * RATE_WINDOW_FRAMES * (RATE_WINDOW_FRAMES ** 2 - 1) / 12.0,
                apparent_closure_from_noise_mps(sigmas=1.0),
                spurious,
                REQUIRED_CLEARANCE_M,
                spurious,
                gap_m,
                spurious,
                gap_m - REQUIRED_CLEARANCE_M,
                worst,
                COMFORT_DECEL_MPS2,
                COMFORT_DECEL_MPS2,
                floor,
            )
        ),
        frames=300,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=20.0,
            accel_fn=lead_constant_speed(),
            label="lead keeping pace at %.0f m" % gap_m,
        ),
        perception=noisy_perception(range_noise_m=HISTORIC_RANGE_NOISE_M, seed=seed),
        expect=Expectation(
            no_collision=True,
            forbid_emergency_intervention=True,
            max_commanded_decel_mps2=COMFORT_DECEL_MPS2,
            max_safety_state=SafetyState.LIMITED,
            min_speed_floor_mps=round(floor, 2),
            min_final_speed_mps=18.0,
            recover_within_frames=120,
        ),
    )


def _noisy_stationary_approach() -> Scenario:
    """A parked car near the avoidability boundary, seen through the same noise.

    The mirror of :func:`_noisy_constant_range` and the reason both are needed:
    a system can be made noise-proof by ignoring the sensor, and this is the
    case that catches it doing so.
    """
    gap_m = 36.0
    stop_m = AVOIDABILITY_M[20.0]
    available = gap_m - stop_m
    needed = 400.0 / (2.0 * (gap_m - REQUIRED_CLEARANCE_M))
    noise3 = apparent_closure_from_noise_mps(sigmas=3.0) / 3.0 * 3.0
    return Scenario(
        name="noisy_stationary_36m_030m_noise",
        summary="parked car 36 m ahead at 20 m/s with +/-0.30 m of range noise",
        guards="missed braking against a stopped obstacle, under sensor noise",
        physics=(
            "The obstacle is stationary and dead ahead at a TRUE 36 m; the ego closes at "
            "20 m/s. Perception reports the range with a seeded zero-mean Gaussian of "
            "standard deviation %.2f m, the same noise as the phantom cases in this family -- "
            "which is the point of having both: noise must not manufacture a brake and must "
            "not suppress one either, and a system can pass the first half by ignoring the "
            "sensor entirely. "
            "The avoidability boundary at 20 m/s -- an emergency-jerk-limited brake committed "
            "at the first honest measurement -- is %.2f m, so %.2f m of clearance is reachable "
            "and the requirement from frame 0 is "
            "400 / (2 x %.0f) = %.2f m/s^2 -- an emergency under EVERY reading of the noisy "
            "range: three standard deviations of noise moves the reported range by %.2f m, "
            "taking the apparent requirement to between %.2f and %.2f m/s^2, all of them above "
            "the %.1f m/s^2 comfort limit. There is no draw of this seed under which the "
            "correct answer is to wait. "
            "WHY IT IS PASSABLE: braking at full authority from frame 0 holds %.2f m, and the "
            "case demands %.1f m, so the system may hesitate for %.0f ms -- %d frames -- and "
            "still pass. TOO PASSIVE: contact, or a true gap below %.1f m. TOO AGGRESSIVE: not "
            "detectable here, where full authority is the only defensible command; the "
            "aggressive direction under noise is carried by the two "
            "noisy_range_* cases above."
            % (
                HISTORIC_RANGE_NOISE_M,
                stop_m,
                available,
                gap_m - REQUIRED_CLEARANCE_M,
                needed,
                3.0 * HISTORIC_RANGE_NOISE_M,
                400.0 / (2.0 * (gap_m + 3.0 * HISTORIC_RANGE_NOISE_M - REQUIRED_CLEARANCE_M)),
                400.0 / (2.0 * (gap_m - 3.0 * HISTORIC_RANGE_NOISE_M - REQUIRED_CLEARANCE_M)),
                COMFORT_DECEL_MPS2,
                available,
                REQUIRED_CLEARANCE_M,
                1000.0 * (available - REQUIRED_CLEARANCE_M) / 20.0,
                int((available - REQUIRED_CLEARANCE_M) / 20.0 / DT_S),
                REQUIRED_CLEARANCE_M,
            )
        ),
        frames=340,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=0.0,
            accel_fn=lead_stationary(),
            label="parked car at 36 m",
        ),
        perception=noisy_perception(range_noise_m=HISTORIC_RANGE_NOISE_M, seed=20240915),
        expect=Expectation(
            no_collision=True,
            min_clearance_m=REQUIRED_CLEARANCE_M,
            must_intervene=True,
            unjustified_brake_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
        ),
    )


def _bias_far_stationary() -> Scenario:
    """A 10 % FAR range bias on the approach to a parked car.

    A bias is not noise and cannot be treated as noise: it does not average out,
    a corroboration window is blind to it, and every frame agrees with every
    other frame about a range that is wrong.
    """
    true_gap = 36.0
    bias = 0.10
    reported = true_gap * (1.0 + bias)
    stop_m = AVOIDABILITY_M[20.0]
    reported_need = 400.0 / (2.0 * (reported - REQUIRED_CLEARANCE_M))
    return Scenario(
        name="range_bias_far_10pct_stationary_36m",
        summary="parked car at a TRUE 36 m reported at 39.6 m: a 10 % far range bias",
        guards="missed braking under a systematic range error",
        physics=(
            "A mono camera's range is f x H / h. Get the height prior H wrong by 10 %% -- a "
            "saloon prior applied to an SUV -- and every range is 10 %% wrong, in the same "
            "direction, for the whole run. Here the obstacle is stationary at a TRUE %.0f m "
            "and is reported at %.1f m on every frame. Noise averages out and a corroboration "
            "window sees through it; a bias does neither, which is why it needs its own case. "
            "WHAT THE SYSTEM SEES: a stationary obstacle at %.1f m closing at 20 m/s. Holding "
            "the %.1f m clearance from %.1f m needs 400 / (2 x %.1f) = %.2f m/s^2 -- an "
            "emergency by the REPORTED scene alone, on the very first frame. A system that "
            "trusts its range completely is therefore still obliged to brake immediately, and "
            "this case does not ask it to guess that it is being lied to. "
            "WHAT ACTUALLY HAPPENS: the avoidability boundary at 20 m/s, with the emergency jerk "
            "limit and the two-frame measurement floor both charged, is %.2f m, so from the true "
            "%.0f m the reachable clearance is %.2f m and the case requires "
            "%.1f m -- %.2f m of headroom, or %d frames of hesitation. "
            "TOO PASSIVE: contact, or a true gap below %.1f m; a system that waits until the "
            "REPORTED range reaches its own trigger point discovers that the reported "
            "avoidability boundary, %.2f m, is a true %.2f m, which is already past the point "
            "of no return -- that is exactly the failure this case exists to catch. "
            "TOO AGGRESSIVE: not detectable at this range, where the requirement is %.2f m/s^2 "
            "and full authority is defensible; the aggressive direction under a range error is "
            "carried by range_bias_near_10pct_constant_range_47m."
            % (
                true_gap,
                reported,
                reported,
                REQUIRED_CLEARANCE_M,
                reported,
                reported - REQUIRED_CLEARANCE_M,
                reported_need,
                stop_m,
                true_gap,
                true_gap - stop_m,
                REQUIRED_CLEARANCE_M,
                true_gap - stop_m - REQUIRED_CLEARANCE_M,
                int((true_gap - stop_m - REQUIRED_CLEARANCE_M) / 20.0 / DT_S),
                REQUIRED_CLEARANCE_M,
                stop_m,
                stop_m / (1.0 + bias),
                400.0 / (2.0 * (true_gap - REQUIRED_CLEARANCE_M)),
            )
        ),
        frames=340,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=true_gap,
            initial_speed_mps=0.0,
            accel_fn=lead_stationary(),
            label="parked car at a true 36 m",
        ),
        perception=PerceptionSpec(range_bias_frac=bias),
        expect=Expectation(
            no_collision=True,
            min_clearance_m=REQUIRED_CLEARANCE_M,
            must_intervene=True,
            unjustified_brake_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
        ),
    )


def _bias_near_constant_range() -> Scenario:
    """A 10 % NEAR range bias that drags a safe follow inside the phantom region.

    The aggressive direction of the same defect, and it is placed exactly where
    it bites: the true gap is OUTSIDE the measured 25e3ba5 phantom region and
    the reported gap is INSIDE it, so the only thing that can make a system fire
    here is the bias.
    """
    true_gap = 47.0
    bias = -0.10
    reported = true_gap * (1.0 + bias)
    floor = _gap_opening_speed_floor_mps(20.0, reported)
    return Scenario(
        name="range_bias_near_10pct_constant_range_47m",
        summary="matched-speed follow at a TRUE 47 m reported at 42.3 m: a 10 %% near bias",
        guards="round-1 phantom AEB (systematic range error)",
        physics=(
            "Both vehicles hold 20 m/s at a constant TRUE gap of %.0f m for 15 s, so the true "
            "closing rate is identically zero, the true requirement is identically zero, and "
            "%.0f m is OUTSIDE the measured 25e3ba5 phantom region, which ends at 43 m at "
            "20 m/s. A -10 %% range bias -- an SUV prior applied to a saloon -- reports that "
            "gap as %.1f m on every frame, which is INSIDE it. The reported range is CONSTANT: "
            "a multiplicative bias on a constant range is still a constant, so the reported "
            "closing rate is exactly zero too and there is no closure anywhere in the "
            "measurements, biased or not. "
            "The only way to fire here is to brake on RANGE ALONE with no closure, which is "
            "the round-1 phantom in its purest form; and the only way to be misled by the bias "
            "is to have a range threshold that fires without one. "
            "WHY IT IS PASSABLE: the reported gap %.1f m is still inside the vehicle's own "
            "%.0f m policy spacing, so opening the gap is correct behaviour -- but opening it "
            "is headway keeping, bounded by the %.1f m/s^2 comfort limit, and opening %.1f m "
            "at comfort and recovering at the plant's 2.5 m/s^2 costs at most %.2f m/s. "
            "TOO AGGRESSIVE: emergency authority, a command above %.1f m/s^2, a state worse "
            "than LIMITED, or a speed floor below %.1f m/s. TOO PASSIVE: finishing below "
            "18 m/s, or never returning to NOMINAL -- the bias never goes away, so a system "
            "that latches on it never lets go."
            % (
                true_gap,
                true_gap,
                reported,
                reported,
                POLICY_GAP_20MPS_M,
                COMFORT_DECEL_MPS2,
                POLICY_GAP_20MPS_M - reported,
                20.0 - floor,
                COMFORT_DECEL_MPS2,
                floor,
            )
        ),
        frames=300,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=true_gap,
            initial_speed_mps=20.0,
            accel_fn=lead_constant_speed(),
            label="lead keeping pace at a true 47 m",
        ),
        perception=PerceptionSpec(range_bias_frac=bias),
        expect=Expectation(
            no_collision=True,
            forbid_emergency_intervention=True,
            max_commanded_decel_mps2=COMFORT_DECEL_MPS2,
            max_safety_state=SafetyState.LIMITED,
            min_speed_floor_mps=round(floor, 2),
            min_final_speed_mps=18.0,
            recover_within_frames=120,
        ),
    )


#: The three overrunning frames used by the frame-overrun scenarios.
#:
#: ``0.1877 s`` is the production log's own worst frame ("Frame 325 took
#: 187.7 ms, more than 3x the 50.0 ms nominal period"); ``0.1743 s`` is the
#: maximum interval measured on this board's 200-frame run; ``0.10 s`` is a
#: dropped frame, which is what a drop looks like to the consumer -- a doubled
#: period, not a gap.  Together they steal
#: ``(0.1877 - 0.05) + (0.1743 - 0.05) + (0.10 - 0.05) = 0.362 s`` of decision
#: time, which at 20 m/s is 7.24 m of travel with nothing being decided.
OVERRUN_FRAMES: Dict[int, float] = {40: 0.1877, 70: 0.1743, 100: 0.10}

OVERRUN_LOST_S = sum(v - DT_S for v in OVERRUN_FRAMES.values())
"""Decision time the overruns steal, seconds."""


def _frame_overrun_stationary() -> Scenario:
    """A stopped obstacle approached through three measured frame overruns."""
    gap_m = 40.0
    stop_m = STOP_DISTANCE_M[20.0]
    lost_m = OVERRUN_LOST_S * 20.0
    worst_clearance = gap_m - stop_m - lost_m
    return Scenario(
        name="frame_overrun_during_stationary_approach",
        summary="parked car 40 m ahead at 20 m/s with three measured frame overruns",
        guards="frame overrun (the 187.7 ms frame in the production log)",
        physics=(
            "A parked car sits 40 m ahead and the ego closes at 20 m/s. Three frames of the "
            "run do not take the nominal 50 ms: frame %d takes %.4f s (the production log's "
            "own 'Frame 325 took 187.7 ms, more than 3x the 50.0 ms nominal period'), frame %d "
            "takes %.4f s (the maximum interval measured on this board's 200-frame run) and "
            "frame %d takes %.2f s (a dropped frame, which to the consumer is a doubled period "
            "rather than a gap). "
            "A long frame is not a pause: the world keeps moving and the last command keeps "
            "acting. These three steal %.3f s of decision time, %.2f m of travel at 20 m/s. "
            "WHY IT IS PASSABLE: full authority through the plant stops the ego in %.2f m, so "
            "%.2f m of clearance is available from 40 m; even a system that decided NOTHING "
            "during all three overruns arrives with %.2f - %.2f = %.2f m, and this case "
            "demands %.1f m. The requirement from frame 0 is only 400 / (2 x 38) = %.2f m/s^2, "
            "below the %.1f m/s^2 comfort limit, so an ordinary firm deceleration is the "
            "correct answer and there is no need to be quick about it. "
            "TOO PASSIVE: contact, or a true gap below %.1f m -- which is what a controller "
            "that integrates a nominal 50 ms while %.0f ms elapsed produces, because its "
            "deceleration arrives a third of a second late. TOO AGGRESSIVE: braking beyond the "
            "oracle's justified ceiling of %.2f x 1.5 + 0.5 = %.2f m/s^2 -- a long frame is a "
            "computer problem and not a hazard, and a system that reads a 187 ms interval as a "
            "sudden closure has invented one."
            % (
                40, OVERRUN_FRAMES[40], 70, OVERRUN_FRAMES[70], 100, OVERRUN_FRAMES[100],
                OVERRUN_LOST_S,
                lost_m,
                stop_m,
                gap_m - stop_m,
                gap_m - stop_m,
                lost_m,
                worst_clearance,
                REQUIRED_CLEARANCE_M,
                400.0 / (2.0 * (gap_m - REQUIRED_CLEARANCE_M)),
                COMFORT_DECEL_MPS2,
                REQUIRED_CLEARANCE_M,
                1000.0 * max(OVERRUN_FRAMES.values()),
                400.0 / (2.0 * (gap_m - REQUIRED_CLEARANCE_M)),
                400.0 / (2.0 * (gap_m - REQUIRED_CLEARANCE_M)) * 1.5 + 0.5,
            )
        ),
        frames=340,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=0.0,
            accel_fn=lead_stationary(),
            label="parked car at 40 m",
        ),
        config=PlantConfig(dt_schedule=dt_with_overruns(340, OVERRUN_FRAMES)),
        expect=Expectation(
            no_collision=True,
            min_clearance_m=REQUIRED_CLEARANCE_M,
            must_intervene=True,
            unjustified_brake_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
        ),
    )


def _jittering_dt_constant_range() -> Scenario:
    """A constant-range follow at a frame period that is never twice the same.

    Placed at 40 m, INSIDE the measured 25e3ba5 phantom region (which ends at
    43 m at 20 m/s), so a system that fires here fails for the reason the
    scenario names rather than for being far away.
    """
    gap_m = 40.0
    schedule = _jitter_schedule(300)
    floor = _gap_opening_speed_floor_mps(20.0, gap_m)
    return Scenario(
        name="variable_dt_constant_range_40m",
        summary="matched-speed follow at 20 m/s and 40 m at a frame period jittering 50-125 ms",
        guards="round-1 phantom AEB (a rate estimator that assumes a fixed period)",
        physics=(
            "Both vehicles hold 20 m/s at a constant 40 m, so the true closing rate is "
            "identically zero whatever the frame period is. The period is not the nominal "
            "50 ms on any frame: it cycles through %s s, spanning 20 Hz down to 8 Hz, which "
            "brackets the interval actually measured on this board (median 58.56 ms, p95 "
            "75.06 ms, maximum 174.33 ms). "
            "A closing rate is a range difference divided by an elapsed time. An estimator "
            "that divides by the NOMINAL period when 125 ms elapsed inflates every rate by "
            "2.5x; one that divides by the elapsed time is exact. Here the true range "
            "difference is exactly zero on every interval, so 2.5 times zero is still zero -- "
            "which is precisely what makes this case decisive: a system that fires cannot "
            "blame the timing, because there is no rate to scale. It has fabricated the rate "
            "from something other than the range. "
            "WHY IT IS PASSABLE: the gap is inside the vehicle's own %.0f m policy spacing, so "
            "opening it is correct and is bounded by the %.1f m/s^2 comfort limit; opening "
            "%.0f m at comfort and recovering at the plant's 2.5 m/s^2 costs at most %.2f m/s. "
            "TOO AGGRESSIVE: emergency authority, a command above %.1f m/s^2, a state worse "
            "than LIMITED, or a speed floor below %.1f m/s. TOO PASSIVE: finishing below "
            "18 m/s, or a degraded state that outlives a disturbance that never existed."
            % (
                ", ".join("%.3f" % v for v in _JITTER_CYCLE),
                POLICY_GAP_20MPS_M,
                COMFORT_DECEL_MPS2,
                POLICY_GAP_20MPS_M - gap_m,
                20.0 - floor,
                COMFORT_DECEL_MPS2,
                floor,
            )
        ),
        frames=300,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=20.0,
            accel_fn=lead_constant_speed(),
            label="lead keeping pace at 40 m",
        ),
        config=PlantConfig(dt_schedule=schedule),
        expect=Expectation(
            no_collision=True,
            forbid_emergency_intervention=True,
            max_commanded_decel_mps2=COMFORT_DECEL_MPS2,
            max_safety_state=SafetyState.LIMITED,
            min_speed_floor_mps=round(floor, 2),
            min_final_speed_mps=18.0,
            recover_within_frames=120,
        ),
    )


#: The repeating frame-period pattern used by the variable-dt scenario, seconds.
#:
#: 20 Hz, 16 Hz, 13.3 Hz, 11.4 Hz, 10 Hz and 8 Hz.  Deterministic and short
#: enough to quote in full in a physics string, and it brackets the intervals
#: this board actually measured (median 58.56 ms, p95 75.06 ms, max 174.33 ms).
_JITTER_CYCLE = (0.050, 0.0625, 0.075, 0.0875, 0.100, 0.125)


def _jitter_schedule(frames: int) -> tuple:
    """``_JITTER_CYCLE`` repeated to cover ``frames`` frames."""
    return tuple(_JITTER_CYCLE[i % len(_JITTER_CYCLE)] for i in range(int(frames)))


def _hostile_sensor_constant_range() -> Scenario:
    """Every sensor imperfection at once, on the phantom boundary.

    Range noise, lateral noise, box jitter and ego-speed noise do not occur one
    at a time on the road, and a rate estimator that survives each of them
    separately can still be broken by their sum: box jitter enters the range
    through the pinhole, so it adds to the range noise before either is
    differentiated.  This case is also the only one that puts noise and the
    production tracker in the loop together.
    """
    gap_m = 40.0
    box_px = 0.3
    # A 1.5 m car at 40 m subtends f*H/d = 900*1.5/40 = 33.75 px, so a jitter of
    # b px is a fractional range error of b/33.75 and a range error of
    # 40*b/33.75 metres.
    box_range_m = gap_m * box_px / (900.0 * 1.5 / gap_m)
    total_sigma = math.sqrt(HISTORIC_RANGE_NOISE_M ** 2 + box_range_m ** 2)
    spurious = apparent_closure_from_noise_mps(noise_m=total_sigma)
    worst = _decel_for_closure(spurious, gap_m)
    floor = _gap_opening_speed_floor_mps(20.0, gap_m)
    return Scenario(
        name="hostile_sensor_constant_range_40m",
        summary="matched-speed follow at 20 m/s and 40 m with every sensor channel noisy",
        guards="round-1 phantom AEB (all noise channels at once, real tracker)",
        physics=(
            "The world is constant_range_40m: both vehicles hold 20 m/s at a constant 40 m, "
            "true closing rate identically zero, true requirement identically zero on all 300 "
            "frames, and 40 m is INSIDE the measured 25e3ba5 phantom region (which ends at "
            "43 m at 20 m/s). Every sensor channel is noisy at once and the production "
            "MultiObjectTracker is in the loop: range %.2f m, box corners %.1f px, lateral "
            "%.2f m, ego speed %.2f m/s. "
            "The channels are not independent of each other and that is the point of "
            "combining them: with a real tracker the range comes FROM the box, so %.1f px of "
            "corner jitter on a %.1f m car at %.0f m -- which subtends %.1f px -- is another "
            "%.2f m of range error, and it adds in quadrature to the %.2f m of range noise for "
            "a total of %.2f m. "
            "WHY IT IS PASSABLE: a least-squares slope over the %d-sample window then has "
            "standard error %.2f / sqrt(%.4f) = %.2f m/s, so three standard errors is a "
            "spurious closure of %.2f m/s, and holding the %.1f m clearance against that at "
            "%.0f m needs %.2f m/s^2 -- still below the %.1f m/s^2 comfort limit. Even a "
            "system that believed the noisiest possible reading of every channel "
            "simultaneously is not entitled to emergency authority here. "
            "TOO AGGRESSIVE: emergency authority, a command above %.1f m/s^2, a state worse "
            "than LIMITED, or a speed floor below %.1f m/s. TOO PASSIVE: finishing below "
            "18 m/s."
            % (
                HISTORIC_RANGE_NOISE_M, box_px, 0.15, 0.20,
                box_px, 1.5, gap_m, 900.0 * 1.5 / gap_m, box_range_m,
                HISTORIC_RANGE_NOISE_M, total_sigma,
                RATE_WINDOW_FRAMES, total_sigma,
                DT_S * DT_S * RATE_WINDOW_FRAMES * (RATE_WINDOW_FRAMES ** 2 - 1) / 12.0,
                apparent_closure_from_noise_mps(noise_m=total_sigma, sigmas=1.0),
                spurious, REQUIRED_CLEARANCE_M, gap_m, worst, COMFORT_DECEL_MPS2,
                COMFORT_DECEL_MPS2, floor,
            )
        ),
        frames=300,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=20.0,
            accel_fn=lead_constant_speed(),
            label="lead keeping pace at 40 m",
        ),
        perception=noisy_perception(
            range_noise_m=HISTORIC_RANGE_NOISE_M,
            seed=20250214,
            lateral_noise_m=0.15,
            box_noise_px=box_px,
            ego_speed_noise_mps=0.20,
            use_real_tracker=True,
        ),
        expect=Expectation(
            no_collision=True,
            forbid_emergency_intervention=True,
            max_commanded_decel_mps2=COMFORT_DECEL_MPS2,
            max_safety_state=SafetyState.LIMITED,
            min_speed_floor_mps=round(floor, 2),
            min_final_speed_mps=18.0,
            recover_within_frames=120,
        ),
    )


def _uncalibrated_camera_stationary() -> Scenario:
    """A parked car dead ahead with NO camera calibration.

    README records this board as "Real, UNCALIBRATED by default", so the
    uncalibrated path is not an exotic configuration -- it is the shipping one,
    and until this case existed no scenario ran it.
    """
    gap_m = 40.0
    boundary = AVOIDABILITY_M[20.0]
    needed = 400.0 / (2.0 * (gap_m - REQUIRED_CLEARANCE_M))
    return Scenario(
        name="uncalibrated_camera_stationary_40m",
        summary="parked car 40 m ahead at 20 m/s with no camera homography",
        guards="missed braking with no metric lane geometry",
        physics=(
            "The obstacle is stationary and dead ahead at 40 m and the ego closes at 20 m/s, "
            "so the requirement is 400 / (2 x %.0f) = %.2f m/s^2 and the avoidability boundary "
            "is %.2f m, leaving %.2f m of reachable clearance against the %.1f m demanded -- "
            "%.2f s of decision slack. Kinematically this is the same case as "
            "stationary_20mps_at_40m. "
            "The difference is that the camera has NO homography. There is no metric lane "
            "model, no ground-plane boundary polynomial for the tracker's corridor test to "
            "read, and no published ego lateral offset; the lane is a pixel column and nothing "
            "else. README.md records this board as 'Real, UNCALIBRATED by default', so this is "
            "not an exotic configuration but the shipping one, and no scenario ran it before. "
            "LOSING THE LANE MUST NOT LOSE THE OBSTACLE. A range is a range whether or not the "
            "lane is metric: the box is still there, the pinhole still divides by its height, "
            "and the closing rate is still the derivative of the range. A system that only "
            "brakes when it can prove which lane the obstacle is in has made a lane failure "
            "into a braking failure, which is the round-1 blocker with a different trigger. "
            "TOO PASSIVE: contact, or a true gap below %.1f m. TOO AGGRESSIVE: braking beyond "
            "the oracle's justified ceiling of %.2f x 1.5 + 0.5 = %.2f m/s^2 for more than the "
            "%d-frame window -- an uncalibrated camera is not an emergency, it is a "
            "degradation, and the correct response to it is the ordinary one."
            % (
                gap_m - REQUIRED_CLEARANCE_M, needed, boundary, gap_m - boundary,
                REQUIRED_CLEARANCE_M, (gap_m - boundary - REQUIRED_CLEARANCE_M) / 20.0,
                REQUIRED_CLEARANCE_M, needed, needed * 1.5 + 0.5,
                JUSTIFICATION_WINDOW_FRAMES,
            )
        ),
        frames=340,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=0.0,
            accel_fn=lead_stationary(),
            label="parked car at 40 m",
        ),
        perception=PerceptionSpec(camera_calibrated=False),
        expect=Expectation(
            no_collision=True,
            min_clearance_m=REQUIRED_CLEARANCE_M,
            must_intervene=True,
            unjustified_brake_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
        ),
    )


# --------------------------------------------------------------------------- #
# Family 5: laterality -- the in-path gate, in both directions
# --------------------------------------------------------------------------- #
#
# The round-1 blocker called "lane-error blindness" has two halves and the
# library only ever tested one of them.  ``lane_error_with_closing_lead`` asks
# that a broken lane estimate must not HIDE a hazard.  These ask the other half:
# a vehicle that is not in the ego's path must not become one, and a lane
# estimate that is wrong towards the next lane must not manufacture a hazard out
# of a car that was never in the way.
#
# Both directions are needed because each is trivially satisfiable on its own.
# A system that brakes for everything passes the first; a system that brakes for
# nothing passes the second.

OUT_OF_LANE_CLEARANCE_M = LANE_WIDTH_M - (LANE_WIDTH_M / 2.0) - 0.9
"""Lateral clearance between a next-lane car's footprint and the ego lane edge.

One lane width from the true centre, minus half a lane, minus half a 1.8 m
car: 0.85 m.  Close enough that a lane estimate wrong by a metre puts the car
back inside the reported lane, which is exactly what
``out_of_lane_vehicle_dragged_in_by_lane_error`` does on purpose.
"""


def _out_of_lane_matched_speed() -> Scenario:
    """A car in the NEXT LANE, 12 m ahead, matching the ego's speed.

    The direct test of the in-path gate, and it is decisive because of where it
    is placed: 12 m is far inside the vehicle's own 52 m policy spacing, so a
    system that cannot tell which lane the car is in will treat it as a lead and
    open the gap to 52 m -- 40 m of gap opening, costing several m/s of speed
    for the rest of the run. A system that CAN tell holds 20 m/s and does
    nothing at all. The two behaviours are metres per second apart, which is why
    this case does not need the car to be closing in order to discriminate.

    It is also oracle-safe by construction.  The harness's kinematic oracle is
    longitudinal: it grades against the gap to the scenario's lead and knows
    nothing about lateral position.  A next-lane car that CLOSES would therefore
    be scored as an unbraked hazard by the oracle itself, and the scenario would
    be demanding the wrong thing.  Matched speed removes that: the true closing
    rate is exactly zero, the oracle's requirement is exactly zero, and the only
    question left is whether the system brakes for a car beside it.
    """
    gap_m = 12.0
    return Scenario(
        name="out_of_lane_vehicle_at_12m",
        summary="a car one lane over, 12 m ahead, matching the ego's 20 m/s",
        guards="round-1 lane-error blindness (the in-path gate, aggressive direction)",
        physics=(
            "A car sits one full lane width (%.1f m) right of the true lane centre, 12 m "
            "ahead, holding the ego's 20 m/s for the whole 15 s run. Its footprint clears the "
            "ego lane edge by %.2f m and clears the ego's own 1.8 m body by "
            "%.1f - 0.9 - 0.9 = %.1f m, so no part of it is ever on a collision course: the "
            "ego could drive past it at any speed and the two would not touch. The true "
            "closing rate is identically zero as well, so the longitudinal requirement is "
            "identically zero on all 300 frames from two independent directions. "
            "The correct response is therefore NOTHING -- not a gentle deceleration, not a "
            "headway correction, not a LIMITED state. "
            "WHAT A LATERALLY BLIND SYSTEM DOES, which is what this case measures: it reads a "
            "vehicle 12 m ahead, compares that against the vehicle's own %.0f m policy spacing "
            "(12 m + 2.0 s x 20 m/s) and opens the gap by %.0f m, which costs "
            "sqrt(%.0f / (1/6 + 1/5)) = %.1f m/s of speed at the comfort limit and leaves the "
            "ego trailing an empty lane. That is not a subtle difference in a metric; it is a "
            "different journey. "
            "TOO AGGRESSIVE: any commanded deceleration above 0.5 m/s^2, any state above "
            "NOMINAL, or any speed below %.1f m/s. TOO PASSIVE: not detectable here and "
            "deliberately so -- there is nothing to react to. The passive direction for the "
            "in-path gate is carried by lane_error_with_closing_lead, where a broken lane "
            "estimate must NOT hide a car that really is dead ahead."
            % (
                LANE_WIDTH_M,
                OUT_OF_LANE_CLEARANCE_M,
                LANE_WIDTH_M,
                LANE_WIDTH_M - 1.8,
                POLICY_GAP_20MPS_M,
                POLICY_GAP_20MPS_M - gap_m,
                POLICY_GAP_20MPS_M - gap_m,
                20.0 - _gap_opening_speed_floor_mps(20.0, gap_m),
                19.5,
            )
        ),
        frames=300,
        ego_speed_mps=20.0,
        lead=adjacent_lane_object(
            initial_gap_m=gap_m,
            speed_mps=20.0,
            label="car in the next lane at 12 m",
        ),
        expect=Expectation(
            no_collision=True,
            forbid_emergency_intervention=True,
            max_commanded_decel_mps2=0.5,
            max_safety_state=SafetyState.NOMINAL,
            min_speed_floor_mps=19.5,
            min_final_speed_mps=19.5,
        ),
    )


def _out_of_lane_dragged_in_by_lane_error() -> Scenario:
    """The same car, with the lane estimate wrong by a metre towards it.

    The round-1 lane-error blindness with the sign reversed: a lane estimator
    that fails TOWARDS the next lane reports a car that is not in the way as
    ``in_ego_lane = True``, and a system that believes that flag brakes for
    traffic beside it on every bend where the lane fit slips.
    """
    gap_m = 12.0
    error_m = 0.86
    return Scenario(
        name="out_of_lane_vehicle_dragged_in_by_lane_error",
        summary="next-lane car 12 m ahead while the lane estimate is 0.86 m wrong towards it",
        guards="round-1 lane-error blindness (a lane error must not CREATE a hazard)",
        physics=(
            "The world is exactly out_of_lane_vehicle_at_12m: a car one lane width (%.1f m) "
            "right of the true lane centre, 12 m ahead, matching the ego's 20 m/s, true "
            "closing rate identically zero. The difference is the lane estimator, which "
            "believes the lane centre is %.2f m to the right of where it really is. "
            "That is exactly the threshold, and it is chosen rather than rounded: a %.1f m "
            "car at %.1f m needs the estimate to be wrong by %.2f m before its footprint "
            "overlaps the REPORTED ego lane, so at %.2f m the perception stack publishes "
            "in_ego_lane = True for a vehicle that is not in the way, and publishes a lane "
            "model whose polynomials agree with it. Both the flag and the geometry are "
            "consistently, confidently wrong -- which is the only honest way to model a lane "
            "fit that has slipped. The error is not made larger than the threshold on "
            "purpose: the ego follows the centre it believes in, so a bigger error steers it "
            "further towards the next lane, and past about 0.9 m the two footprints really do "
            "overlap and 'do not brake' would stop being the correct answer. Measured here, "
            "the ego settles %.2f m off its true lane centre and the clear distance to the "
            "other car never falls below %.2f m. "
            "WHY IT IS PASSABLE: the box the detector produces is unaffected. A box is "
            "projected from the object's position relative to the EGO, not relative to the "
            "lane, so the car's image-space centre is still %.0f px off the image centre at "
            "12 m and an in-path gate that works in image space still sees it outside the "
            "corridor. The arbiter's own documented contract says the same thing in words: it "
            "does not read TrackedObject.in_ego_lane, and a lane model may only WIDEN its "
            "corridor, never move it. This case is that sentence made executable. "
            "TOO AGGRESSIVE: any commanded deceleration above 0.5 m/s^2, any state above "
            "NOMINAL, or any speed below %.1f m/s -- braking for a car in the next lane "
            "because the lane fit slipped a metre. TOO PASSIVE: not detectable here; the "
            "passive direction is carried by lane_error_with_closing_lead."
            % (
                LANE_WIDTH_M,
                error_m,
                1.8,
                LANE_WIDTH_M,
                LANE_WIDTH_M - (LANE_WIDTH_M / 2.0) - 0.9,
                error_m,
                1.62,
                LANE_WIDTH_M - 1.62 - 0.9 - 0.9,
                LANE_WIDTH_M * 900.0 / 12.0,
                19.5,
            )
        ),
        frames=300,
        ego_speed_mps=20.0,
        lead=adjacent_lane_object(
            initial_gap_m=gap_m,
            speed_mps=20.0,
            label="car in the next lane at 12 m",
        ),
        perception=PerceptionSpec(lane_offset_error_m=error_m),
        expect=Expectation(
            no_collision=True,
            forbid_emergency_intervention=True,
            max_commanded_decel_mps2=0.5,
            max_safety_state=SafetyState.NOMINAL,
            min_speed_floor_mps=19.5,
            min_final_speed_mps=19.5,
            # The self-protecting half. The ego follows the lane centre it
            # believes in, so it drifts towards the other car; measured, it
            # settles 1.62 m off the TRUE centre, inside the 1.75 m half-lane.
            # If a change ever drifts it further the two footprints overlap,
            # this case stops being "do not brake for a car beside you" and
            # starts being "do not brake for a car you are steering into" --
            # and lane_departure fires here rather than the scenario quietly
            # demanding the wrong thing.
            max_abs_lateral_offset_m=1.75,
        ),
    )


# --------------------------------------------------------------------------- #
# Family 6: MANY objects at once -- blocked on a plumbing change
# --------------------------------------------------------------------------- #

MULTI_OBJECT_FIELD = "others"
"""The :class:`~tests.scenarios.scenario.Scenario` field these cases need."""


def scenario_supports_multi_object() -> bool:
    """Whether :class:`Scenario` can carry more than one object yet.

    :class:`tests.scenarios.plant.Plant` has accepted ``others`` and published
    ``WorldState.objects``, ``WorldState.in_path_objects`` and
    ``WorldState.min_in_path_gap_m`` for two revisions, and nothing can reach
    them: ``Scenario`` has no field for them and
    :func:`tests.scenarios.scenario.run` builds the plant with a lead and
    nothing else.  A capability the specification cannot express is not
    coverage.

    This probe exists so that the two scenarios below appear the moment the
    field does, rather than waiting for somebody to remember them, and so that
    :func:`tests.test_scenarios` can fail with a specific message meanwhile
    instead of the library quietly being two cases short.
    """
    return MULTI_OBJECT_FIELD in getattr(Scenario, "__dataclass_fields__", {})


MULTI_OBJECT_BLOCKED_REASON = (
    "tests/scenarios/scenario.py does not plumb multiple objects through yet: "
    "Scenario has no '%s' field and run() builds Plant(lead=...) with no others, "
    "so WorldState.objects can never hold more than the lead. The plant, the "
    "sensor and WorldState.min_in_path_gap_m are all ready; what is missing is "
    "(1) 'others: Tuple[LeadSpec, ...] = ()' on Scenario, (2) passing it to "
    "Plant(others=scenario.others) in run(), and (3) an oracle that grades "
    "against WorldState.min_in_path_gap_m rather than WorldState.gap_m, so that "
    "an out-of-lane object is not scored as an unbraked longitudinal hazard. "
    "Until (3) lands, an in-path judgement over several objects is not "
    "expressible and the simultaneous case below would be graded wrongly rather "
    "than merely missing." % MULTI_OBJECT_FIELD
)
"""Why :func:`multi_object_scenarios` is empty, quoted verbatim by the tests."""


def multi_object_scenarios() -> List[Scenario]:
    """The two cases that need more than one object in the world.

    Empty until :func:`scenario_supports_multi_object` is true.  They are
    written out in full rather than left as a comment so that the plumbing
    change is a one-line diff in ``scenario.py`` and nothing here, and so that
    the coverage census can count what is missing rather than what was
    forgotten.
    """
    if not scenario_supports_multi_object():
        return []
    kwargs = {}

    # 1. An out-of-lane vehicle CLOSING, with nothing in the ego lane at all.
    #    Unlike out_of_lane_vehicle_at_12m this one is overtaken, so a laterally
    #    blind system sees 8 m/s of closure onto a 40 m gap and brakes hard.  It
    #    needs the multi-object oracle: with no in-path object the correct
    #    minimum in-path gap is +inf, and the lead-based oracle would score the
    #    overtaken car as an unbraked hazard.
    kwargs[MULTI_OBJECT_FIELD] = (
        adjacent_lane_object(
            initial_gap_m=40.0,
            speed_mps=12.0,
            label="slower car in the next lane, overtaken",
        ),
    )
    overtake = Scenario(
        name="out_of_lane_vehicle_overtaken",
        summary="ego at 20 m/s overtakes a 12 m/s car in the next lane; ego lane empty",
        guards="round-1 lane-error blindness (the in-path gate under closure)",
        physics=(
            "The ego lane is EMPTY for the whole run. A car one lane width right of the "
            "centre, 40 m ahead, holds 12 m/s while the ego holds 20 m/s, so the ego closes "
            "on it at 8 m/s, draws level at t = 5.0 s and leaves it behind. Its footprint "
            "clears the ego's by %.1f m at every instant, so the minimum IN-PATH gap is "
            "infinite for all 300 frames and the collision-avoidance requirement is "
            "identically zero. "
            "This is the strongest form of the in-path test and the one that needs the "
            "multi-object oracle: the closure is real, large and sustained, and the only "
            "thing that makes braking wrong is the lateral offset. A system that ignores "
            "laterality reads 8 m/s onto a shrinking 40 m gap and goes to emergency authority "
            "somewhere around 13 m, i.e. it brakes hard for every vehicle it ever overtakes. "
            "TOO AGGRESSIVE: any emergency authority, any deceleration above 0.5 m/s^2, or "
            "any speed below 19.5 m/s. TOO PASSIVE: not applicable; there is no hazard."
            % (LANE_WIDTH_M - 1.8,)
        ),
        frames=300,
        ego_speed_mps=20.0,
        lead=None,
        expect=Expectation(
            no_collision=True,
            forbid_emergency_intervention=True,
            max_commanded_decel_mps2=0.5,
            max_safety_state=SafetyState.NOMINAL,
            min_speed_floor_mps=19.5,
            min_final_speed_mps=19.5,
        ),
        **kwargs
    )

    # 2. Both at once: the real hazard is in the lane, the decoy is beside it.
    kwargs2 = {
        MULTI_OBJECT_FIELD: (
            adjacent_lane_object(
                initial_gap_m=18.0,
                speed_mps=20.0,
                label="car in the next lane at 18 m",
            ),
        )
    }
    simultaneous = Scenario(
        name="in_lane_and_out_of_lane_hazards",
        summary="lead braking at 6 m/s^2 at 26 m while a next-lane car sits 18 m ahead",
        guards="round-2 missed braking with an out-of-lane distractor",
        physics=(
            "Two objects. The one that matters is IN the ego lane at 26 m, matched at 20 m/s, "
            "and brakes at 6 m/s^2 from frame 0 -- the upper edge of the measured 1ce4886 "
            "contact region (16-26 m at 20 m/s), so this is a case a missed brake fails. The "
            "other is one lane width right at 18 m, matching speed, and must be ignored: it is "
            "NEARER, so a system that selects its lead by range alone selects the wrong "
            "object, and the wrong object is not decelerating. "
            "The lead stops after 3.33 s having covered 33.3 m, so the ego has "
            "26 + 33.3 - 2 = 57.3 m in which to stop from 20 m/s against the %.2f m the plant "
            "needs at full authority; the intervention may begin as late as %.2f s after the "
            "lead's brake light. The distractor's minimum in-path gap is infinite throughout. "
            "TOO PASSIVE: contact, or a minimum in-path gap under 2.0 m -- which is what "
            "picking the nearest object rather than the nearest IN-PATH object produces, and "
            "it is the exact shape of the round-1 defect wearing a round-2 consequence. "
            "TOO AGGRESSIVE: braking beyond the oracle's justified ceiling for more than the "
            "%d-frame window, or reacting to the next-lane car at all before the lead's "
            "deceleration is measurable."
            % (
                STOP_DISTANCE_M[20.0],
                (26.0 + 33.33 - REQUIRED_CLEARANCE_M - STOP_DISTANCE_M[20.0]) / 20.0,
                JUSTIFICATION_WINDOW_FRAMES,
            )
        ),
        frames=300,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=26.0,
            initial_speed_mps=20.0,
            accel_fn=lead_brakes(6.0, start_s=0.0),
            label="lead braking at 6 m/s^2 from frame 0",
        ),
        expect=Expectation(
            no_collision=True,
            min_clearance_m=REQUIRED_CLEARANCE_M,
            must_intervene=True,
            unjustified_brake_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
        ),
        **kwargs2
    )
    return [overtake, simultaneous]


def _reid_during_closing_approach() -> Scenario:
    """A re-identification in the middle of a REAL closure.

    The case ``reid_during_steady_follow`` was renamed away from.  On a
    steady follow the true closing rate is zero, and a new track with NO rate
    also reports zero, so the two are indistinguishable: whatever an estimator
    does with a fresh track, the number it produces is the number the truth
    produces, and the scenario cannot fail for the reason it names.  Measured:
    on ``25e3ba5`` that scenario reported exactly the same five diagnoses as
    ``constant_range_40m`` -- ``phantom_intervention``,
    ``forbidden_emergency_brake``, ``forbidden_emergency_state``,
    ``over_authority_state``, ``over_braked`` -- so it added no discrimination
    to the library at all.

    Here the true closure is 8 m/s, so the three candidate answers a fresh
    track can produce are all different numbers and all have different
    consequences:

    ======================  ===========  ===================================
    what the estimator does  reports      consequence
    ======================  ===========  ===================================
    seeds from ego speed     -20 m/s      demands emergency 58 frames early
    starts with no history     0 m/s      no closure at all: the brake is
                                          deferred past the point of safety
    re-fits from range        -8 m/s      correct
    ======================  ===========  ===================================

    Both failure directions are therefore reachable in ONE scenario, which is
    the property the whole library is being rebuilt around.
    """
    ego = 20.0
    lead_v = 12.0
    closure = ego - lead_v
    gap0 = 60.0
    warrant_gap = closure * closure / (2.0 * COMFORT_DECEL_MPS2) + REQUIRED_CLEARANCE_M
    warrant_t = (gap0 - warrant_gap) / closure
    reid = (60, 90, int(round(warrant_t / DT_S)))
    seeded_gap = gap0 - closure * reid[0] * DT_S
    seeded_demand = (ego * ego) / (2.0 * (seeded_gap - REQUIRED_CLEARANCE_M))
    return Scenario(
        name="reid_during_closing_approach",
        summary="closing on a 12 m/s lead from 60 m at 20 m/s; the tracker re-ids it 3 times",
        guards="round-1 phantom AEB (new-track rate seed) AND round-2 missed braking",
        physics=(
            "The ego holds %.0f m/s and the lead holds %.0f m/s, so the TRUE closing rate is "
            "%.0f m/s and the gap falls from %.0f m at a constant %.0f m/s. Holding the "
            "%.1f m clearance needs %.0f^2 / (2 (g - %.1f)) m/s^2, which reaches the "
            "%.1f m/s^2 comfort limit at g = %.2f m, i.e. at t = %.2f s (frame %d). That is "
            "the first frame at which emergency authority is warranted; before it, ordinary "
            "following control still has the situation. "
            "On frames %d, %d and %d the tracker assigns a NEW id to the same physical "
            "vehicle. A track id is bookkeeping and carries no information about motion, so "
            "nothing physical changes on those frames -- but every per-track estimate "
            "downstream is destroyed, and what an estimator puts in their place is the whole "
            "question. "
            "THE THREE ANSWERS, all different here and all the same on a steady follow, which "
            "is why this case exists and the steady-follow one was renamed: (a) seeding the "
            "new track's rate from the ego speed reports %.0f m/s of closure at a true "
            "%.0f m gap on frame %d and demands %.0f^2 / (2 x %.0f) = %.2f m/s^2, an emergency "
            "%d frames before one is warranted; (b) reporting no rate at all reports 0 m/s and "
            "defers the brake indefinitely; (c) re-fitting from the track's own range history "
            "recovers %.0f m/s within a few frames and is correct. "
            "WHY IT IS PASSABLE: the last re-id lands on the warrant frame itself, and the "
            "plant's estimator needs %d samples to produce any rate at all, so a correct "
            "system is blind for %d frames and re-acquires by frame %d -- against a mandate "
            "that does not arrive until the gap reaches about %.1f m, because full authority "
            "from %.0f m/s against a lead still doing %.0f m/s only has to wash out %.0f m/s "
            "of relative speed, which costs about %.1f m. There is roughly half a second of "
            "slack, and a system may also brake on RANGE ALONE at that distance without "
            "needing a rate. "
            "TOO AGGRESSIVE: braking beyond the oracle's justified ceiling for more than the "
            "%d-frame window -- which is what (a) does on frame %d. TOO PASSIVE: contact, or a "
            "minimum true gap below %.1f m -- which is what (b) does."
            % (
                ego, lead_v, closure, gap0, closure,
                REQUIRED_CLEARANCE_M, closure, REQUIRED_CLEARANCE_M,
                COMFORT_DECEL_MPS2, warrant_gap, warrant_t, reid[2],
                reid[0], reid[1], reid[2],
                ego, seeded_gap, reid[0], ego, seeded_gap - REQUIRED_CLEARANCE_M,
                seeded_demand, reid[2] - reid[0],
                closure,
                2, 2, reid[2] + 2,
                7.0, ego, lead_v, closure, 5.0,
                JUSTIFICATION_WINDOW_FRAMES, reid[0], REQUIRED_CLEARANCE_M,
            )
        ),
        frames=300,
        ego_speed_mps=ego,
        lead=LeadSpec(
            initial_gap_m=gap0,
            initial_speed_mps=lead_v,
            accel_fn=lead_constant_speed(),
            label="lead at 12 m/s, closed on at 8 m/s",
        ),
        perception=PerceptionSpec(reid_frames=reid),
        expect=Expectation(
            no_collision=True,
            min_clearance_m=REQUIRED_CLEARANCE_M,
            must_intervene=True,
            unjustified_brake_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
        ),
    )


def _real_tracker_constant_range() -> Scenario:
    """The phantom boundary with the PRODUCTION tracker in the loop.

    Every other scenario injects :class:`TrackedObject` s straight from the
    sensor model, which is fast and isolates the decision layer.  This one runs
    detections through :class:`adas.tracking.MultiObjectTracker` instead, so the
    Kalman range filter, the Hungarian association, the M-of-N confirmation and
    the tracker's own ``in_ego_lane`` decision all execute -- and the phantom
    AEB lived in exactly that code.  One case, because it costs a Hungarian
    solve per frame; on the boundary, because anywhere else it proves nothing.
    """
    gap_m = 40.0
    floor = _gap_opening_speed_floor_mps(20.0, gap_m)
    return Scenario(
        name="real_tracker_constant_range_40m",
        summary="matched-speed follow at 20 m/s and 40 m through the real MultiObjectTracker",
        guards="round-1 phantom AEB (through the production tracker)",
        physics=(
            "The world is exactly constant_range_40m: both vehicles hold 20 m/s at a constant "
            "40 m for 15 s, the true closing rate is identically zero, the true requirement is "
            "identically zero on all 300 frames, and 40 m is INSIDE the measured 25e3ba5 "
            "phantom region, which ends at 43 m at 20 m/s. "
            "The difference is the path the measurement takes. Every other case in this "
            "library injects tracks from the sensor model; this one renders a detection box "
            "and puts it through adas.tracking.MultiObjectTracker, so the Kalman range filter, "
            "the association, the M-of-N confirmation and the tracker's own ego-lane decision "
            "all run. That matters twice over: the phantom AEB lived in that code, and the "
            "tracker withholds a new object for its first %d frames, which is up to 150 ms of "
            "confirmation latency that the injected path does not have and the road does. "
            "A Kalman filter fed a constant range converges to a zero range rate; it cannot "
            "invent a closure out of a flat measurement, so a correct stack reports "
            "approximately zero closing throughout and this case is passable. "
            "TOO AGGRESSIVE: emergency authority, a command above %.1f m/s^2, a state worse "
            "than LIMITED, or a speed floor below %.1f m/s. TOO PASSIVE: finishing below "
            "18 m/s, or never returning to NOMINAL after a confirmation transient that is over "
            "in %d frames."
            % (3, COMFORT_DECEL_MPS2, floor, 3)
        ),
        frames=300,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=20.0,
            accel_fn=lead_constant_speed(),
            label="lead keeping pace at 40 m",
        ),
        perception=PerceptionSpec(use_real_tracker=True),
        expect=Expectation(
            no_collision=True,
            forbid_emergency_intervention=True,
            max_commanded_decel_mps2=COMFORT_DECEL_MPS2,
            max_safety_state=SafetyState.LIMITED,
            min_speed_floor_mps=round(floor, 2),
            min_final_speed_mps=18.0,
            recover_within_frames=120,
        ),
    )


# --------------------------------------------------------------------------- #
# The library
# --------------------------------------------------------------------------- #


def build() -> List[Scenario]:
    """Return the whole library, in report order.

    Ordered by family so that the report reads as a specification: the phantom
    boundary first, then the two missed-braking boundaries, then the cases that
    are about something other than a boundary.
    """
    scenarios: List[Scenario] = []

    # ------------------------------------------- constant range (phantom) ----
    # Straddles the measured 25e3ba5 phantom boundary at 43/44 m: four cases
    # inside it, two outside.
    scenarios += [
        _constant_range(12.0, inside_phantom_region=True),
        _constant_range(20.0, inside_phantom_region=True),
        _constant_range(30.0, inside_phantom_region=True),
        _constant_range(40.0, inside_phantom_region=True),
        _constant_range(52.0, inside_phantom_region=False),
        _constant_range(70.0, inside_phantom_region=False),
    ]

    # ------------------------------------------ lead braking (missed brake) --
    # Straddles the measured 1ce4886 contact region, d0 = 16-26 m at 20 m/s.
    scenarios += [
        _lead_brakes_hard(
            20.0, 16.0,
            "1ce4886 makes CONTACT at the lower edge of its measured region (minimum gap "
            "-0.00 m); 25e3ba5 clears by only 0.34 m. Contact begins at 16 m and this case is "
            "the first one inside it.",
        ),
        _lead_brakes_hard(
            20.0, 20.0,
            "the middle of 1ce4886's measured contact region: minimum gap -0.65 m against "
            "25e3ba5's +0.40 m. Neither commit reaches the 2.0 m clearance, which is the point "
            "-- a 0.4 m miss is not a pass.",
        ),
        _lead_brakes_hard(
            20.0, 26.0,
            "the upper edge of 1ce4886's measured contact region: minimum gap -0.12 m at 26 m "
            "and +0.28 m at 27 m, so 26 m is the last range at which it collides. 25e3ba5 "
            "clears by 1.56 m.",
        ),
        _lead_brakes_hard(
            20.0, 32.0,
            "just OUTSIDE the contact region: 1ce4886 clears by 2.22 m and 25e3ba5 by 3.11 m, "
            "so neither collides and the case is decided by the 2.0 m clearance and by the "
            "over-braking ceiling instead. It is the sensitivity control for the three cases "
            "above -- a regression that widens the contact region by 6 m shows up here first.",
        ),
        _lead_brakes_hard(
            15.0, 15.0,
            "at 15 m/s neither commit collides, so this case is decided on CLEARANCE alone: "
            "1ce4886 holds 1.55 m and 25e3ba5 holds 3.04 m against the 2.0 m the specification "
            "requires and the 12.99 m the vehicle can hold. It is the case that fails the "
            "round-2 commit without any contact at all.",
        ),
        _lead_brakes_hard(
            25.0, 30.0,
            "BOTH commits collide here -- 1ce4886 by 3.08 m and 25e3ba5 by 1.72 m -- so this "
            "case is the one that neither round of fixes ever passed. 25e3ba5's contact region "
            "at 25 m/s is 25-40 m and 1ce4886's is 16-40 m.",
        ),
    ]

    # ---------------------------------- stationary obstacle (missed brake) ---
    # Placed against STOP_DISTANCE_M: three cases within 1.2 m of the
    # avoidability boundary, one just outside it, one clear, one far.
    scenarios += [
        _stationary(
            15.0, 24.0, 340, REQUIRED_CLEARANCE_M,
            tail_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
            note="RE-PLACED. The avoidability boundary at 15 m/s is 18.77 m once the emergency "
            "jerk limit and the two-frame measurement floor are charged, so a 2.0 m clearance "
            "is first reachable at 21.0 m and this case sits 3.0 m outside that: 5.23 m is "
            "reachable, 2.00 m is demanded, and the difference is 0.20 s -- FOUR frames -- of "
            "decision slack, so the case is on the boundary without being a knife edge. It "
            "used to sit at 15 m demanding 0.25 m, where the best any correct system could "
            "hold was -0.21 m: unsatisfiable, and passable only by braking on the stationary "
            "prior before any range rate existed. Measured at the nearby 22 m and 25 m: "
            "25e3ba5 holds 5.99 and 6.52 m, 1ce4886 holds 5.34 and 7.34 m, both commanding "
            "8.00 m/s^2 against a requirement of 4.86 m/s^2 -- so on this plant the case is "
            "decided on OVER-BRAKING rather than on contact, and that is what the boundary "
            "now looks like.",
        ),
        _stationary(
            20.0, 36.0, 340, REQUIRED_CLEARANCE_M,
            tail_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
            note="RE-PLACED, and this is the tightest satisfiable case in the family: the "
            "boundary at 20 m/s is 31.30 m, a 2.00 m clearance is first reachable at 33.5 m, "
            "and 4.70 m is reachable here -- 0.10 s, TWO frames, of decision slack. It "
            "replaces the 26 m and 29 m cases, which demanded 0.10 m and 2.00 m where -0.30 m "
            "and -0.23 m were reachable, i.e. which no correct system could ever have passed. "
            "Measured: 25e3ba5 holds 7.55 m and 1ce4886 holds 6.54 m.",
        ),
        _stationary(
            25.0, 52.0, 340, REQUIRED_CLEARANCE_M,
            tail_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
            note="RE-PLACED from 42 m, where 1.00 m was demanded and -0.17 m was reachable. The "
            "boundary at 25 m/s is 46.95 m and 2.0 m is first reachable at 49.0 m; 52 m leaves "
            "5.05 m. The highest speed in the family, where the requirement (7.00 m/s^2) is "
            "closest to full authority and there is least room for a late decision. Measured: "
            "25e3ba5 holds 8.03 m, 1ce4886 holds 7.59 m.",
        ),
        _stationary(
            20.0, 50.0, 340, REQUIRED_CLEARANCE_M, over_brake_ceiling_mps2=6.8,
            note="the OVER-BRAKING case at a range where the answer is still an emergency: the "
            "requirement is 400 / (2 x 48) = 4.17 m/s^2, so the oracle's justified ceiling is "
            "4.17 x 1.5 + 0.5 = 6.75 m/s^2 and a compliant system has 18.70 m of clearance to "
            "play with. Both commits go to 8.00 m/s^2 anyway and both fail it. Without this "
            "case the whole family is satisfied by braking flat out at every range, which is "
            "precisely the behaviour that transfers the collision to the vehicle behind.",
        ),
        _stationary(
            20.0, 40.0, 340, REQUIRED_CLEARANCE_M,
            note="the PAIR with the 36 m case, 4 m further out: 8.70 m reachable instead of "
            "4.70 m, 0.30 s of decision slack instead of 0.10 s. Two ranges 4 m apart at the "
            "same speed is what makes a boundary MEASURABLE -- a regression that moves it out "
            "by 3 m fails 36 m and passes 40 m, so the failure has a WIDTH rather than a name. "
            "Both commits stop well clear here: 8.04 m on 25e3ba5, 8.32 m on 1ce4886. It is "
            "also the comfortably-clear control, the case that must keep passing while the "
            "boundary cases fail so a regression can be localised to the boundary rather than "
            "to braking in general.",
        ),
        _stationary(
            20.0, 75.0, 420, REQUIRED_CLEARANCE_M, over_brake_ceiling_mps2=4.6,
            note=(
                "at 75 m the true requirement is 2.74 m/s^2, BELOW the 3.0 m/s^2 comfort limit, "
                "so the correct response is a firm but ordinary deceleration and the oracle's "
                "justified ceiling is 2.74 x 1.5 + 0.5 = 4.6 m/s^2. Both commits go to full "
                "authority anyway -- 6.41 m/s^2 on 25e3ba5 and 5.91 m/s^2 on 1ce4886 -- and "
                "both fail this case on over-braking, which is the failure direction the "
                "stationary family had no way to express before."
            ),
        ),
    ]

    # ------------------------------------------------- gentle lead braking ---
    scenarios.append(
        Scenario(
            name="lead_brakes_gently_2mps2",
            summary="steady follow at 20 m/s and 40 m, then the lead brakes gently at 2 m/s^2",
            guards="the unpoliced sub-emergency band",
            physics=(
                "The lead needs 10.0 s and 100.0 m to stop from 20 m/s at 2 m/s^2. The ego has "
                "40 + 100 - 2 = 138 m in which to stop from 20 m/s, which a constant "
                "400 / (2 x 138) = 1.45 m/s^2 achieves -- less than half the 3.0 m/s^2 comfort "
                "limit. The correct response is therefore a gentle deceleration, and the "
                "oracle's justified ceiling is 1.45 x 1.5 + 0.5 = 2.68 m/s^2. "
                "TOO AGGRESSIVE: anything above the 3.0 m/s^2 comfort limit, which reads a "
                "routine deceleration as a threat; the ceiling is set at comfort rather than "
                "at the 3.5 m/s^2 emergency threshold precisely because the half-metre band "
                "between them is the one the backtest found unpoliced. TOO PASSIVE: contact, "
                "or a minimum gap below 2.0 m, which 138 m of budget makes inexcusable."
            ),
            frames=340,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=40.0,
                initial_speed_mps=20.0,
                accel_fn=lead_brakes(2.0, start_s=0.0),
                label="lead braking at 2 m/s^2",
            ),
            expect=Expectation(
                no_collision=True,
                min_clearance_m=REQUIRED_CLEARANCE_M,
                must_intervene=True,
                max_commanded_decel_mps2=COMFORT_DECEL_MPS2,
            ),
        )
    )

    scenarios.append(
        Scenario(
            name="lead_accelerates_away",
            summary="lead 25 m ahead at 20 m/s accelerates to 28 m/s",
            guards="round-1 phantom AEB",
            physics=(
                "The lead is never slower than the ego and its acceleration is positive "
                "throughout, so the true collision-avoidance requirement is identically zero "
                "for the whole run and the gap opens without the ego doing anything. The gap "
                "starts at 25 m, inside the 52 m policy spacing, so a headway correction "
                "bounded by the 3.0 m/s^2 comfort limit is defensible -- but only until the "
                "gap opens, which at a closing rate that reverses within one second is a "
                "couple of seconds at most. Three m/s^2 for two seconds costs 6 m/s. "
                "TOO AGGRESSIVE: emergency authority, a state worse than LIMITED, more than "
                "3.0 m/s^2 of brake, or dropping below 14 m/s -- all of them braking for a "
                "vehicle that is running away. TOO PASSIVE: failing to return to NOMINAL "
                "within 3 s of the gap opening, which is the degraded state sustaining itself."
            ),
            frames=200,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=25.0,
                initial_speed_mps=20.0,
                accel_fn=lead_accelerates(1.5, cap_mps=28.0),
                label="lead pulling away",
            ),
            expect=Expectation(
                no_collision=True,
                forbid_emergency_intervention=True,
                max_commanded_decel_mps2=COMFORT_DECEL_MPS2,
                max_safety_state=SafetyState.LIMITED,
                min_speed_floor_mps=14.0,
                recover_within_frames=60,
            ),
        )
    )

    # ------------------------------------------------------------- cut-in ---
    scenarios.append(
        Scenario(
            name="cutin_close_at_12m",
            summary="a car doing 12 m/s cuts in 12 m ahead of an ego doing 20 m/s",
            physics=(
                "At the moment of the cut-in the gap is 12 m and the closing rate is 8 m/s. "
                "Holding 2.0 m of clearance permits 10 m of further closure, so the required "
                "constant deceleration is 8^2 / (2 x 10) = 3.2 m/s^2 -- above the 3.0 m/s^2 "
                "comfort limit, so this IS a genuine emergency and an intervention is "
                "mandatory. It is also comfortably survivable: at 8 m/s^2 the relative speed "
                "washes out in 4.0 m. "
                "TOO PASSIVE: contact, or a minimum gap below 0.5 m. TOO AGGRESSIVE: braking "
                "beyond the justified ceiling for more than five frames; the first few frames "
                "of a brand-new track carry no rate history, so a quarter of a second of "
                "over-strong response is a reaction and anything longer is a guess."
            ),
            frames=300,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=20.0,
                initial_speed_mps=12.0,
                accel_fn=lead_constant_speed(),
                appears_at_s=1.0,
                label="cut-in at 12 m",
            ),
            expect=Expectation(
                no_collision=True,
                min_clearance_m=0.5,
                must_intervene=True,
                unjustified_brake_frames_allowed=5,
                recover_within_frames=100,
            ),
        )
    )

    scenarios.append(
        Scenario(
            name="cutin_moderate_at_25m",
            summary="a car doing 14 m/s cuts in 25 m ahead of an ego doing 20 m/s",
            physics=(
                "At the cut-in the gap is 25 m and the closing rate is 6 m/s. Holding 2.0 m of "
                "clearance permits 23 m of closure, so the requirement is "
                "6^2 / (2 x 23) = 0.78 m/s^2 -- below the 1.0 m/s^2 floor at which anything "
                "counts as a hazard at all. "
                "TOO AGGRESSIVE: emergency authority, or more than 3.0 m/s^2 of brake. A "
                "cut-in is not automatically an emergency, and a system that treats every new "
                "in-path track as one will brake hard several times per motorway journey. "
                "TOO PASSIVE: letting the gap fall below 2.0 m, or never returning to NOMINAL "
                "after the closing rate has washed out."
            ),
            frames=250,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=31.0,
                initial_speed_mps=14.0,
                accel_fn=lead_constant_speed(),
                appears_at_s=1.0,
                label="cut-in at 25 m",
            ),
            expect=Expectation(
                no_collision=True,
                min_clearance_m=REQUIRED_CLEARANCE_M,
                forbid_emergency_intervention=True,
                max_commanded_decel_mps2=COMFORT_DECEL_MPS2,
                unjustified_brake_frames_allowed=5,
                recover_within_frames=100,
            ),
        )
    )

    # -------------------------------------------------------- empty road ----
    scenarios.append(
        Scenario(
            name="empty_road_nominal",
            summary="20 s of empty straight road at 15 m/s with a valid ego state",
            guards="round-1 disengagement on an empty road",
            physics=(
                "There is no object, the ego state is valid on every frame and perception is "
                "healthy on every frame. There is no measurement in this run that differs from "
                "the measurement a correctly functioning system sees on a clear road, so any "
                "state other than NOMINAL is generated entirely inside the system. Round 1 "
                "disengaged here. "
                "TOO AGGRESSIVE: any brake at all beyond 0.5 m/s^2, any state above NOMINAL, "
                "or any speed loss below 14 m/s -- on an empty road at a steady cruise speed "
                "the correct brake command is exactly zero. TOO PASSIVE: finishing below "
                "14 m/s, which means a command that was never released."
            ),
            frames=400,
            ego_speed_mps=15.0,
            lead=None,
            expect=Expectation(
                forbid_emergency_intervention=True,
                max_commanded_decel_mps2=0.5,
                max_safety_state=SafetyState.NOMINAL,
                min_speed_floor_mps=14.0,
                min_final_speed_mps=14.0,
            ),
        )
    )

    scenarios.append(
        Scenario(
            name="empty_road_dropout",
            summary="empty road at 15 m/s with a 0.8 s perception dropout at frame 100",
            guards="self-sustaining loop / recovery",
            physics=(
                "Sixteen frames (0.8 s) of blindness on a road that was empty on the frame "
                "before and empty on the frame after. Degrading while blind is correct: the "
                "system cannot know the road stayed empty. Continuing to degrade after "
                "perception returns is not, because from that frame on every measurement is "
                "the measurement of a clear road. A minimum-risk deceleration of 3.5 m/s^2 for "
                "0.8 s costs 2.8 m/s. "
                "TOO AGGRESSIVE: latching DISENGAGE, or finishing below 12 m/s. TOO PASSIVE: "
                "failing to return to NOMINAL within 2 s of the dropout ending, which is the "
                "assertion that catches a degraded state sustaining itself."
            ),
            frames=300,
            ego_speed_mps=15.0,
            lead=None,
            perception=PerceptionSpec(failed_frames=tuple(range(100, 116))),
            expect=Expectation(
                max_safety_state=SafetyState.MIN_RISK_MANEUVER,
                recover_within_frames=40,
                min_final_speed_mps=12.0,
            ),
        )
    )

    scenarios.append(
        Scenario(
            name="dropout_mid_approach",
            summary="approaching a parked car at 15 m/s when the detector misses it for 0.8 s",
            guards="missed braking against a stopped obstacle",
            physics=(
                "A parked car sits 48 m ahead and is detected from frame 0; from frame 20 to "
                "frame 35 the detector returns nothing for it, then it is detected again. The "
                "car did not cease to exist because the detector blinked. Full authority stops "
                "the ego from 15 m/s in 14.70 m, so the avoidability boundary is 14.70 m and "
                "there is 33.3 m of margin at frame 0; at 15 m/s the ego covers 12 m during "
                "the miss, so even a system that coasted through the whole dropout re-acquires "
                "the car at 36 m with 21 m of margin. The requirement from frame 0 is only "
                "225 / (2 x 46) = 2.45 m/s^2, below the 3.0 m/s^2 comfort limit. "
                "TOO PASSIVE: contact, or a gap below 2.0 m -- either means the dropout was "
                "allowed to release an intervention the world had not stopped requiring. TOO "
                "AGGRESSIVE: braking beyond the justified ceiling of 2.45 x 1.5 + 0.5 = "
                "4.2 m/s^2, i.e. treating a 0.8 s detection miss on a car 40 m away as an "
                "emergency."
            ),
            frames=300,
            ego_speed_mps=15.0,
            lead=LeadSpec(
                initial_gap_m=48.0,
                initial_speed_mps=0.0,
                accel_fn=lead_stationary(),
                label="parked car at 48 m",
            ),
            perception=PerceptionSpec(miss_frames=tuple(range(20, 36))),
            expect=Expectation(
                no_collision=True,
                min_clearance_m=REQUIRED_CLEARANCE_M,
                must_intervene=True,
                unjustified_brake_frames_allowed=0,
            ),
        )
    )

    # ----------------------------------------------------- track identity ---
    scenarios.append(
        Scenario(
            name="track_id_churn_during_steady_follow_40m",
            summary="matched-speed follow at 20 m/s and 40 m; the tracker re-ids the lead 3 times",
            guards="round-1 phantom AEB (new-track rate seed)",
            physics=(
                "The lead holds 20 m/s at a constant 40 m for the whole run: the true closing "
                "rate is identically zero and the true requirement is identically zero. On "
                "frames 60, 120 and 180 the tracker assigns a new id to the same physical "
                "vehicle. A track id is bookkeeping; it carries no information about motion, "
                "so nothing physical changed at frame 60 and nothing in the command may change "
                "either. Any estimator that seeds a new track's closing rate at the ego speed "
                "reads -20 m/s of closure at 40 m and demands an emergency brake. "
                "The gap was moved from 52 m to 40 m because 40 m is INSIDE the measured "
                "25e3ba5 phantom region (which ends at 43 m at 20 m/s) and 52 m was outside "
                "it, so at 52 m this case could not fail for the reason it names. "
                "RENAMED, and the gap recorded rather than papered over. This case used to be "
                "called reid_during_steady_follow and to claim it tested a new-track rate "
                "seed. It cannot. On a steady follow the true closing rate is zero, and a "
                "fresh track with NO history also reports zero, so every answer an estimator "
                "can give on a re-id frame is the same number the truth gives and the case is "
                "insensitive to the defect it named. Measured on 25e3ba5 it reported exactly "
                "the same five diagnoses as constant_range_40m -- phantom_intervention, "
                "forbidden_emergency_brake, forbidden_emergency_state, over_authority_state, "
                "over_braked -- i.e. it added no discrimination to the library whatsoever. "
                "WHAT IT ACTUALLY TESTS, and the reason it is kept: identity CHURN. Every "
                "per-track estimate downstream -- the arbiter's own range filter, its "
                "corroboration window, its age and hit counts -- is destroyed three times, and "
                "each new track re-applies whatever prior the system seeds a fresh track with. "
                "The arbiter's is -ego_speed, so this is the phantom boundary at 40 m entered "
                "three extra times from a cold start. "
                "THE DEFECT IT DOES NOT REACH is a re-seeded rate that is WRONG, and it needs "
                "a closure to be visible: see reid_during_closing_approach, which supplies "
                "one. A plant that could seed a new track's reported velocity_mps from the ego "
                "speed instead of leaving it unknown would let this case test the defect "
                "directly; tests/scenarios/plant.py is not this workstream's file and the "
                "request is recorded in the handoff. "
                "TOO AGGRESSIVE: any deceleration above the 3.0 m/s^2 comfort limit, or a "
                "state worse than LIMITED. TOO PASSIVE: finishing below 18 m/s."
            ),
            frames=300,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=40.0,
                initial_speed_mps=20.0,
                accel_fn=lead_constant_speed(),
                label="lead keeping pace at 40 m",
            ),
            perception=PerceptionSpec(reid_frames=(60, 120, 180)),
            expect=Expectation(
                no_collision=True,
                forbid_emergency_intervention=True,
                max_commanded_decel_mps2=COMFORT_DECEL_MPS2,
                max_safety_state=SafetyState.LIMITED,
                min_speed_floor_mps=round(_gap_opening_speed_floor_mps(20.0, 40.0), 2),
                min_final_speed_mps=18.0,
                recover_within_frames=120,
            ),
        )
    )

    scenarios.append(_reid_during_closing_approach())

    scenarios.append(
        Scenario(
            name="range_jump_during_steady_follow",
            summary="matched-speed follow at 20 m/s and 52 m; the reported range steps 10 m closer",
            guards="round-1 phantom AEB (range re-seed)",
            physics=(
                "The true gap is 52 m for the whole run and the true closing rate is zero. At "
                "frame 60 the reported range steps from 52 m to 42 m in one 50 ms frame and "
                "stays there. Differentiating that step gives 200 m/s of closure, which no "
                "road vehicle can produce; the only physical reading of a 10 m step in 50 ms "
                "is that the measurement changed, not the world. The apparent 42 m lands "
                "inside the measured 25e3ba5 phantom region (gaps up to 43 m at 20 m/s), so a "
                "system that trusts the step will fire here. "
                "TOO AGGRESSIVE: braking as though 200 m/s of closure were real -- anything "
                "above the 3.0 m/s^2 comfort limit, a state worse than LIMITED, or dropping "
                "below 15 m/s. TOO PASSIVE: never returning to NOMINAL once the apparent range "
                "has been steady for several seconds."
            ),
            frames=300,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=52.0,
                initial_speed_mps=20.0,
                accel_fn=lead_constant_speed(),
                label="lead keeping pace",
            ),
            perception=PerceptionSpec(range_jump_at=(60, -10.0)),
            expect=Expectation(
                no_collision=True,
                forbid_emergency_intervention=True,
                max_commanded_decel_mps2=COMFORT_DECEL_MPS2,
                max_safety_state=SafetyState.LIMITED,
                min_speed_floor_mps=15.0,
                recover_within_frames=100,
            ),
        )
    )

    # ---------------------------------------------------------- lane error --
    scenarios.append(
        Scenario(
            name="lane_error_with_closing_lead",
            summary="a mock lane centre 460 px off while a parked car sits 36 m dead ahead",
            guards="round-1 lane-error blindness",
            physics=(
                "The lane estimate is mock, its confidence is zero and it places the lane "
                "centre 460 px left of the image centre. The obstacle is real, stationary, "
                "dead ahead at 36 m, and the ego closes on it at 20 m/s. The avoidability "
                "boundary at 20 m/s is 31.30 m once the emergency jerk limit and the two-frame "
                "measurement floor are charged, so 4.70 m of clearance is reachable, this case "
                "demands 2.00 m of it, and the requirement is 400 / (2 x 34) = 5.88 m/s^2 -- a "
                "genuine emergency with a real but small margin. The range was moved OUT from "
                "30 m, where the best reachable clearance was -0.01 m and the 2.00 m demanded "
                "was therefore unsatisfiable; and IN from the original 65 m, where 39 m of "
                "margin meant a system could ignore the obstacle for two full seconds and "
                "still pass. Both directions of mis-placement, in one scenario's history. "
                "A lane estimate may WIDEN the corridor a hazard is searched in -- more lane "
                "means more places a threat can be -- but it must never move or narrow it, "
                "because a lane detector failing towards the kerb would then hide a car "
                "directly in front of the bumper. TOO PASSIVE: contact, or a gap below 2.0 m; "
                "a collision here is a lateral fault causing a longitudinal one. TOO "
                "AGGRESSIVE: not detectable at this range, where full authority is required; "
                "the aggressive direction for lane faults is carried by empty_road_nominal."
            ),
            frames=300,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=36.0,
                initial_speed_mps=0.0,
                accel_fn=lead_stationary(),
                label="parked car at 36 m",
            ),
            perception=PerceptionSpec(
                lane_center_bias_px=-460.0, lane_is_mock=True, lane_confidence=0.0
            ),
            expect=Expectation(
                no_collision=True,
                min_clearance_m=REQUIRED_CLEARANCE_M,
                must_intervene=True,
                unjustified_brake_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
            ),
        )
    )

    # -------------------------------------------------- sensor fidelity ----
    # Noise, bias and a frame period that is not 20 Hz.  Every one of these
    # features existed in the plant and was used by nothing; see
    # coverage_census() at the bottom of this module, which now counts them.
    scenarios += [
        _noisy_constant_range(20.0, seed=20240914),
        _noisy_constant_range(40.0, seed=20250131),
        _noisy_stationary_approach(),
        _bias_far_stationary(),
        _bias_near_constant_range(),
        _frame_overrun_stationary(),
        _jittering_dt_constant_range(),
        _real_tracker_constant_range(),
        _hostile_sensor_constant_range(),
        _uncalibrated_camera_stationary(),
    ]

    # -------------------------------------------------------- laterality ----
    scenarios += [
        _out_of_lane_matched_speed(),
        _out_of_lane_dragged_in_by_lane_error(),
    ]
    scenarios += multi_object_scenarios()

    # --------------------------------------------------------------- bends --
    scenarios.append(
        Scenario(
            name="bend_nominal",
            summary="15 s on an empty 230 m radius left bend at 20 m/s",
            guards="self-sustaining loop (steering_rate -> plan_accel)",
            physics=(
                "A 230 m radius at 20 m/s is 400 / 230 = 1.74 m/s^2 of lateral acceleration, "
                "well inside the 3.0 m/s^2 a passenger car uses without complaint, and the "
                "road-wheel angle it needs is atan(2.8 / 230) = 0.70 degrees held steady -- a "
                "steering RATE of zero once established. Driving a constant bend is the most "
                "ordinary thing a lane-keeping system does. The road is empty, so there is no "
                "longitudinal hazard of any kind. "
                "TOO AGGRESSIVE: any state above NOMINAL, or more than 0.5 m/s^2 of brake, or "
                "dropping below 18 m/s. TOO PASSIVE: leaving the lane -- 1.75 m either side of "
                "the centre is half of a 3.5 m lane -- which is what a lateral controller that "
                "will not commit to the bend does."
            ),
            frames=300,
            ego_speed_mps=20.0,
            lead=None,
            road=constant_bend(230.0),
            expect=Expectation(
                forbid_emergency_intervention=True,
                max_commanded_decel_mps2=0.5,
                max_safety_state=SafetyState.NOMINAL,
                min_speed_floor_mps=18.0,
                max_abs_lateral_offset_m=1.75,
            ),
        )
    )

    scenarios.append(
        Scenario(
            name="bend_mrm_and_recover",
            summary="empty 400 m radius bend at 20 m/s with a 1.5 s perception dropout",
            guards="held-steering loop in MIN_RISK_MANEUVER",
            physics=(
                "Perception fails for 30 frames (1.5 s) on a 400 m radius bend, which at "
                "20 m/s is 1.0 m/s^2 of lateral acceleration. A second and a half is the "
                "timescale of a transient occlusion -- a wiper pass, a tunnel mouth, a truck's "
                "shadow -- and 30 m of travel; treating it as a permanent loss of the sensor "
                "is a false diagnosis. A minimum-risk manoeuvre must also bring the vehicle to "
                "rest IN ITS OWN LANE: a vehicle that holds the wheel straight while the road "
                "bends deviates by s^2 / (2R), which over the 27 m it covers while "
                "decelerating at 3.5 m/s^2 for 1.5 s is 27^2 / 800 = 0.9 m, comfortably inside "
                "the 1.75 m half-lane. "
                "TOO AGGRESSIVE: latching DISENGAGE, or never returning to NOMINAL within 3 s "
                "of perception coming back on an empty road. TOO PASSIVE: leaving the lane "
                "while blind, which a held or oscillating steering command does."
            ),
            frames=300,
            ego_speed_mps=20.0,
            lead=None,
            road=constant_bend(400.0),
            perception=PerceptionSpec(failed_frames=tuple(range(60, 90))),
            expect=Expectation(
                max_safety_state=SafetyState.MIN_RISK_MANEUVER,
                recover_within_frames=60,
                max_abs_lateral_offset_m=1.75,
            ),
        )
    )

    # ------------------------------------------------- loss of the source ---
    scenarios.append(
        Scenario(
            name="source_loss_mid_run",
            summary="the camera stops delivering permanently at frame 60, ego at 20 m/s",
            guards="ADAS-DEC-21 / minimum-risk stop",
            physics=(
                "From frame 60 the sensor is gone for good and the vehicle is moving at 20 m/s "
                "with no way to see anything. There is exactly one defensible end state: "
                "stopped. A minimum-risk manoeuvre is a controlled stop at about 3.5 m/s^2, "
                "which from 20 m/s takes 5.7 s and 57 m; the run gives it 9 s. "
                "TOO PASSIVE: continuing to cruise blind, i.e. finishing above 1 m/s. TOO "
                "AGGRESSIVE: latching DISENGAGE, which hands a moving vehicle to nobody -- "
                "there is no driver in this loop to catch it -- or exceeding 5.0 m/s^2, which "
                "separates a controlled stop from an emergency stop; nothing was seen, so "
                "nothing warrants full authority."
            ),
            frames=240,
            ego_speed_mps=20.0,
            lead=None,
            perception=PerceptionSpec(source_lost_from_frame=60),
            expect=Expectation(
                max_safety_state=SafetyState.MIN_RISK_MANEUVER,
                max_commanded_decel_mps2=5.0,
                max_final_speed_mps=1.0,
            ),
        )
    )

    scenarios.append(
        Scenario(
            name="terminate_during_emergency",
            summary="the process exits cleanly 2.5 s in, while an intervention is active",
            guards="ADAS-DEC-21 latched command at exit",
            physics=(
                "The lead brakes at 6 m/s^2 from frame 0 at a 20 m gap -- the middle of the "
                "measured contact region -- so at t = 2.5 s the ego is mid-intervention with "
                "the brake applied. The run then ends: a clean exit, a bounded frame count, a "
                "SIGTERM; the reason does not matter. Leaving a loop is not a command, and "
                "whatever was last written to the actuators stays latched there. "
                "TOO PASSIVE: emitting nothing, or emitting a neutral command, either of which "
                "releases the brake of a vehicle still closing on a decelerating car. TOO "
                "AGGRESSIVE: an exit command that still applies throttle, which is the same "
                "latching bug with the opposite sign."
            ),
            frames=50,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=20.0,
                initial_speed_mps=20.0,
                accel_fn=lead_brakes(6.0, start_s=0.0),
                label="lead braking at 6 m/s^2 from frame 0",
            ),
            terminate_during_run=True,
            expect=Expectation(
                no_collision=True,
                require_fail_safe_exit=True,
            ),
        )
    )

    # ------------------------------------------------------------ recovery --
    scenarios.append(
        Scenario(
            name="lead_brakes_then_leaves_lane",
            summary="lead brakes 4 m/s^2 for 1.5 s, then changes lane at t=8 s",
            guards="self-sustaining loop / recovery",
            physics=(
                "From a 30 m follow at 20 m/s the lead sheds 6 m/s over 1.5 s. The ego must "
                "close the speed difference within the 30 m it has, which a deceleration of "
                "well under comfort achieves, so this is an ordinary following manoeuvre with "
                "a real transient in the middle of it. At t=8 s the lead leaves the lane and "
                "the road ahead is empty. From that point every measurement is the measurement "
                "of a clear road. "
                "TOO PASSIVE: not being in NOMINAL within 2 s of the lane change, or finishing "
                "below 12 m/s. All three of the self-sustaining loops in this module's history "
                "-- steering rate feeding plan acceleration feeding lane departure -- fail "
                "exactly here and nowhere else in this library, because this is the only case "
                "that asks the system to let go. TOO AGGRESSIVE: contact or a gap below 1.0 m "
                "during the transient, or a state worse than MIN_RISK_MANEUVER."
            ),
            frames=400,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=30.0,
                initial_speed_mps=20.0,
                accel_fn=lead_brake_then_release(
                    4.0, start_s=1.0, duration_s=1.5, resume_accel_mps2=0.0, resume_to_mps=14.0
                ),
                vanishes_at_s=8.0,
                label="lead brakes then changes lane",
            ),
            expect=Expectation(
                no_collision=True,
                min_clearance_m=1.0,
                must_intervene=True,
                max_safety_state=SafetyState.MIN_RISK_MANEUVER,
                recover_within_frames=40,
                min_final_speed_mps=12.0,
            ),
        )
    )

    return scenarios


SCENARIOS: List[Scenario] = build()
"""The library, built once at import time.  Scenarios are frozen dataclasses,
so sharing one instance between the pytest run and a manual report run is safe.
"""


# --------------------------------------------------------------------------- #
# The coverage census
# --------------------------------------------------------------------------- #


def _uses_out_of_lane_object(scenario: Scenario) -> bool:
    """Whether any object in this scenario sits outside the ego's own corridor.

    Lateral offset is checked against the ego footprint plus the object's own
    half width, which is the plant's ``in_ego_path`` test: an object further out
    than that can never be hit, however hard the ego closes on it.
    """
    specs = [scenario.lead] if scenario.lead is not None else []
    specs += list(getattr(scenario, MULTI_OBJECT_FIELD, ()) or ())
    for spec in specs:
        if spec is None:
            continue
        offsets = [spec.lateral_at(0.0), spec.lateral_at(scenario.frames * DT_S)]
        if any(abs(o) > 0.9 + spec.width_m / 2.0 for o in offsets):
            return True
    return False


def _dt_values(scenario: Scenario) -> List[float]:
    """Every distinct frame period this scenario's plant will actually use."""
    cfg = scenario.config
    used = set(cfg.dt_schedule[: scenario.frames])
    if scenario.frames > len(cfg.dt_schedule):
        used.add(cfg.dt_s)
    return sorted(used)


#: Every fidelity feature the plant offers, and how to tell whether a scenario
#: uses it.  A feature is "used" only when the scenario would behave differently
#: without it -- ``range_noise_m = 0.0`` does not count as noise coverage.
#:
#: This list is the census, and the census is the point: the previous revision
#: of this library grew a plant with sensor noise, range bias, variable frame
#: periods and multi-object worlds in it, and used NONE of them.  Nothing in the
#: repository said so, because nothing counted.  Now something counts, and
#: ``tests/test_scenarios.py::test_harness_fidelity_features_are_exercised``
#: fails when a count that used to be non-zero goes back to zero.
_CENSUS_FEATURES = (
    ("range_noise_m > 0", lambda s: s.perception.range_noise_m > 0.0),
    ("range_bias_frac != 0", lambda s: s.perception.range_bias_frac != 0.0),
    ("lateral_noise_m > 0", lambda s: s.perception.lateral_noise_m > 0.0),
    ("box_noise_px > 0", lambda s: s.perception.box_noise_px > 0.0),
    ("ego_speed_noise_mps > 0", lambda s: s.perception.ego_speed_noise_mps > 0.0),
    ("dt != 0.05 s", lambda s: any(abs(v - DT_S) > 1e-12 for v in _dt_values(s))),
    ("frame overrun schedule", lambda s: bool(s.config.dt_schedule)),
    ("multiple objects", lambda s: bool(getattr(s, MULTI_OBJECT_FIELD, ()))),
    ("out-of-lane object", _uses_out_of_lane_object),
    ("use_real_tracker", lambda s: s.perception.use_real_tracker),
    ("reid_frames", lambda s: bool(s.perception.reid_frames)),
    ("range_jump_at", lambda s: s.perception.range_jump_at is not None),
    ("miss_frames", lambda s: bool(s.perception.miss_frames)),
    ("failed_frames", lambda s: bool(s.perception.failed_frames)),
    ("source_lost_from_frame", lambda s: s.perception.source_lost_from_frame is not None),
    ("lane_offset_error_m != 0", lambda s: s.perception.lane_offset_error_m != 0.0),
    ("lane_center_bias_px != 0", lambda s: s.perception.lane_center_bias_px != 0.0),
    ("lane_is_mock", lambda s: s.perception.lane_is_mock),
    ("uncalibrated camera", lambda s: not s.perception.camera_calibrated),
    ("curved road", lambda s: s.road.label != "straight"),
    ("terminate_during_run", lambda s: s.terminate_during_run),
    ("non-default stack", lambda s: s.stack != DEFAULT_STACK),
)


def coverage_census(scenarios: Optional[List[Scenario]] = None) -> Dict[str, List[str]]:
    """Which scenarios exercise which plant fidelity feature.

    Args:
        scenarios: The corpus to count; defaults to :data:`SCENARIOS`.

    Returns:
        Feature name -> the names of the scenarios that use it, in library
        order.  A feature with an empty list is a plant capability that the
        specification cannot reach, which is the defect this function exists to
        make visible.
    """
    corpus = SCENARIOS if scenarios is None else scenarios
    out: Dict[str, List[str]] = {}
    for label, predicate in _CENSUS_FEATURES:
        out[label] = [s.name for s in corpus if predicate(s)]
    return out


def render_census(scenarios: Optional[List[Scenario]] = None) -> str:
    """The census as a printable table, for the report and for a failure message."""
    corpus = SCENARIOS if scenarios is None else scenarios
    census = coverage_census(corpus)
    width = max(len(k) for k in census)
    lines = ["coverage census over %d scenario(s):" % len(corpus)]
    for label in census:
        names = census[label]
        lines.append(
            "  %-*s %2d/%d  %s"
            % (
                width,
                label,
                len(names),
                len(corpus),
                ", ".join(names) if names else "NOTHING EXERCISES THIS",
            )
        )
    return "\n".join(lines)


def by_name(name: str) -> Scenario:
    """Look up one scenario, for interactive debugging."""
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(name)
