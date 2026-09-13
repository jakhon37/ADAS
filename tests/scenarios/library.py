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

from typing import List, Optional

from adas.core.models import SafetyState

from tests.scenarios.plant import (
    LeadSpec,
    PerceptionSpec,
    lead_accelerates,
    lead_brake_then_release,
    lead_brakes,
    lead_constant_speed,
    lead_stationary,
)
from tests.scenarios.plant import constant_bend
from tests.scenarios.scenario import Expectation, Scenario

# --------------------------------------------------------------------------- #
# Shared arithmetic, quoted in the physics strings below
# --------------------------------------------------------------------------- #

STOP_DISTANCE_M = {10.0: 6.67, 15.0: 14.70, 20.0: 25.85, 25.0: 40.13}
"""Distance to standstill under full authority THROUGH THE PLANT, metres.

Measured by driving :class:`tests.scenarios.plant.Plant` at ``brake = 1.0`` from
each speed, so it includes the 0.15 s brake rise; the idealised ``v^2 / 2a``
figures are 6.25 / 14.06 / 25.00 / 39.06 m.  These four numbers are the
avoidability boundary for a stationary obstacle: at 20 m/s a parked car 25 m
ahead cannot be avoided by any system whatever, and one 26 m ahead can be
avoided by exactly 0.15 m.  Every scenario in the stationary family is placed
relative to this table, which is why none of them is mis-specified as a
"missed brake" when the arithmetic had already decided the outcome.
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
    best_m = gap_m - stop_m
    needed = (ego_mps * ego_mps) / (2.0 * max(0.01, gap_m - REQUIRED_CLEARANCE_M))
    return Scenario(
        name="stationary_%02dmps_at_%02dm" % (ego_mps, gap_m),
        summary="stopped vehicle %.0f m ahead, ego at %.0f m/s on an empty straight"
        % (gap_m, ego_mps),
        guards="missed braking against a stopped obstacle",
        physics=(
            "The obstacle is stationary and dead ahead, so the closing rate is exactly the ego "
            "speed and never changes sign: there is no reading of the measurements under which "
            "doing nothing is correct. Full authority through the plant, 8 m/s^2 behind a "
            "0.15 s brake rise, stops the ego from %.0f m/s in %.2f m, so the best clearance "
            "physically available from %.0f m is %.2f m and the avoidability boundary at this "
            "speed is %.2f m -- a scenario placed closer than that would be asking the system "
            "to beat arithmetic. A constant %.2f m/s^2 from frame 0 is enough to stop with the "
            "%.1f m standstill clearance intact. "
            "MEASURED: %s "
            "TOO PASSIVE: contact, or a minimum true gap below the %.1f m this case requires. "
            "%s"
            % (
                ego_mps,
                stop_m,
                gap_m,
                best_m,
                stop_m,
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
            15.0, 15.0, 340, 0.25,
            tail_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
            note="the avoidability boundary at 15 m/s is 14.70 m, so 15 m is the closest range at "
            "which any system can still stop -- by 0.30 m. 1ce4886 CONTACTS here (-0.43 m) "
            "while 25e3ba5 stops with 0.30 m, exactly the physical limit.",
        ),
        _stationary(
            20.0, 26.0, 340, 0.10,
            tail_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
            note="the avoidability boundary at 20 m/s is 25.85 m. 1ce4886 CONTACTS at both 26 m "
            "(-1.21 m) and 27 m (-0.21 m) although 0.15 m and 1.15 m were available; 25e3ba5 "
            "stops with 0.15 m, the whole of what the vehicle has.",
        ),
        _stationary(
            20.0, 29.0, 340, REQUIRED_CLEARANCE_M,
            tail_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
            note="3.15 m of clearance is available, so the full 2.0 m standstill clearance is "
            "required and the case is decided on margin rather than contact: 1ce4886 holds "
            "1.79 m and FAILS, 25e3ba5 holds 3.15 m and passes. This is the first range "
            "outside 1ce4886's contact region (26-27 m) and the pair with the 26 m case.",
        ),
        _stationary(
            25.0, 42.0, 340, 1.0,
            tail_frames_allowed=JUSTIFICATION_WINDOW_FRAMES,
            note="the avoidability boundary at 25 m/s is 40.13 m, leaving 1.87 m at 42 m. 1ce4886 "
            "holds 0.17 m and 25e3ba5 holds 1.87 m -- the exact physical maximum -- so a 1.0 m "
            "requirement separates them without demanding more than the vehicle has.",
        ),
        _stationary(
            20.0, 40.0, 340, REQUIRED_CLEARANCE_M,
            note="14.15 m of clearance is available and both commits stop, 1ce4886 with 5.90 m and "
            "25e3ba5 with 5.89 m. It is the comfortably-clear control: a case that must keep "
            "passing while the boundary cases fail, so that a regression can be localised to "
            "the boundary rather than to braking in general.",
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
            name="reid_during_steady_follow",
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
                "CAVEAT, and it is the reason this case is currently weaker than it reads: the "
                "plant resets track_id, age and hits on a re-id frame but still reports the "
                "TRUE closing rate, so the specific defect named here -- a re-seeded rate -- "
                "cannot yet occur. What it does test is the phantom boundary at 40 m. "
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
            summary="a mock lane centre 460 px off while a parked car sits 30 m dead ahead",
            guards="round-1 lane-error blindness",
            physics=(
                "The lane estimate is mock, its confidence is zero and it places the lane "
                "centre 460 px left of the image centre. The obstacle is real, stationary, "
                "dead ahead at 30 m, and the ego closes on it at 20 m/s. Full authority stops "
                "the ego in 25.85 m, so 4.15 m of clearance is available and the requirement "
                "is 400 / (2 x 28) = 7.14 m/s^2 -- a genuine emergency with a real but small "
                "margin. The range was moved in from 65 m, where 39 m of margin meant a system "
                "could ignore the obstacle for two full seconds and still pass. "
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
                initial_gap_m=30.0,
                initial_speed_mps=0.0,
                accel_fn=lead_stationary(),
                label="parked car at 30 m",
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


def by_name(name: str) -> Scenario:
    """Look up one scenario, for interactive debugging."""
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(name)
