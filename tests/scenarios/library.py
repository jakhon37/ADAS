"""The scenario library: the executable form of the safety specification.

Every entry states, in its ``physics`` field, the arithmetic that makes its
expected outcome the correct one.  Nothing here was derived by running the
current arbiter and writing down what it did; the numbers come from stopping
distances, closing rates and the authority limits declared in
:mod:`tests.scenarios.plant`.

Coverage is organised so that each historical failure has a named guard:

``round-1 phantom AEB``
    :func:`constant_range_motorway`, :func:`reid_during_steady_follow`,
    :func:`range_jump_during_steady_follow`, :func:`empty_road_nominal`
``round-1 disengagement on an empty road``
    :func:`empty_road_nominal`, :func:`empty_road_dropout`
``round-1 lane-error blindness``
    :func:`lane_error_with_closing_lead`
``round-2 missed braking against a lead braking at 6 m/s^2``
    the four :func:`lead_brakes_hard` cases at 15, 20, 25 and 30 m
``self-sustaining loops (steering_rate -> plan_accel -> lane_departure)``
    every scenario carrying ``recover_within_frames``
``ADAS-DEC-21 latched command at exit``
    :func:`terminate_during_emergency`, :func:`source_loss_mid_run`
"""

from __future__ import annotations

from typing import List

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
#
#   Full-authority stopping distance at 8 m/s^2 with the 0.15 s brake rise
#   (equivalent dead time ~0.065 s):
#
#       10 m/s ->  6.9 m      20 m/s -> 26.3 m
#       15 m/s -> 15.0 m      25 m/s -> 40.7 m
#
#   A lead braking from 20 m/s at 6 m/s^2 stops in 3.33 s and covers 33.3 m.
#   A lead braking from 20 m/s at 2 m/s^2 stops in 10.0 s and covers 100.0 m.
#
#   The vehicle's own spacing policy is d0 + tau*v with d0 = 12 m and
#   tau = 2.0 s, so at 20 m/s it asks for 52 m.  A gap at or above the policy
#   distance gives the longitudinal law nothing to correct.


def _stationary(
    name: str, ego_mps: float, gap_m: float, frames: int, stop_distance_m: float
) -> Scenario:
    """A parked vehicle in the ego lane, visible from the first frame."""
    return Scenario(
        name=name,
        summary="stationary vehicle %.0f m ahead, ego at %.0f m/s on an empty straight"
        % (gap_m, ego_mps),
        physics=(
            "Full-authority braking at 8 m/s^2 with the 0.15 s brake rise stops the ego from "
            "%.0f m/s in %.1f m. Adding the 2.0 m standstill clearance the specification "
            "requires gives %.1f m, against a %.0f m gap, so contact is avoidable with "
            "%.1f m to spare. The obstacle is stationary and dead ahead: the closing rate is "
            "exactly the ego speed and never changes sign, so there is no interpretation of "
            "the measurements under which doing nothing is correct. Failing to stop here is a "
            "missed brake; stopping is the only acceptable outcome."
            % (ego_mps, stop_distance_m, stop_distance_m + 2.0, gap_m, gap_m - stop_distance_m - 2.0)
        ),
        frames=frames,
        ego_speed_mps=ego_mps,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=0.0,
            accel_fn=lead_stationary(),
            label="parked car",
        ),
        expect=Expectation(
            no_collision=True,
            min_clearance_m=0.5,
            must_intervene=True,
        ),
    )


def _lead_brakes_hard(gap_m: float) -> Scenario:
    """The round-2 regression: a lead braking at 6 m/s^2 from a steady follow."""
    lead_stop_m = 20.0 * 20.0 / (2.0 * 6.0)
    budget = gap_m + lead_stop_m - 2.0
    return Scenario(
        name="lead_brakes_6mps2_from_%02dm" % gap_m,
        summary="steady follow at 20 m/s and %.0f m, then the lead brakes at 6 m/s^2 to a stop"
        % gap_m,
        guards="round-2 missed braking",
        physics=(
            "Both vehicles run at 20 m/s for the first second, so the closing rate is zero and "
            "no braking is warranted yet. From t=1.0 s the lead decelerates at 6 m/s^2, "
            "stopping after 3.33 s having covered %.1f m. The ego therefore has "
            "%.0f + %.1f - 2.0 = %.1f m in which to stop from 20 m/s, and full authority needs "
            "26.3 m. The intervention may begin as late as (%.1f - 26.3) / 20 = %.2f s after "
            "the lead starts braking and still avoid contact with the 2 m clearance intact. "
            "This is the exact case the previous commit collided on, from all four of "
            "15, 20, 25 and 30 m, where its predecessor had stopped with 3.5-10.0 m to spare."
            % (lead_stop_m, gap_m, lead_stop_m, budget, budget, (budget - 26.3) / 20.0)
        ),
        frames=250,
        ego_speed_mps=20.0,
        lead=LeadSpec(
            initial_gap_m=gap_m,
            initial_speed_mps=20.0,
            accel_fn=lead_brakes(6.0, start_s=1.0),
            label="lead braking at 6 m/s^2",
        ),
        expect=Expectation(
            no_collision=True,
            min_clearance_m=0.5,
            must_intervene=True,
        ),
    )


def build() -> List[Scenario]:
    """Return the whole library, in report order."""
    scenarios: List[Scenario] = []

    # ---------------------------------------------------------- stationary --
    scenarios += [
        _stationary("stationary_10mps_at_25m", 10.0, 25.0, 200, 6.9),
        _stationary("stationary_15mps_at_45m", 15.0, 45.0, 250, 15.0),
        _stationary("stationary_20mps_at_60m", 20.0, 60.0, 250, 26.3),
        _stationary("stationary_25mps_at_75m", 25.0, 75.0, 300, 40.7),
    ]

    # ------------------------------------------------------ lead braking ----
    scenarios += [_lead_brakes_hard(g) for g in (15.0, 20.0, 25.0, 30.0)]

    scenarios.append(
        Scenario(
            name="lead_brakes_gently_2mps2",
            summary="steady follow at 20 m/s and 40 m, then the lead brakes gently at 2 m/s^2",
            physics=(
                "The lead needs 10.0 s and 100.0 m to stop from 20 m/s at 2 m/s^2. The ego has "
                "40 + 100 - 2 = 138 m in which to stop from 20 m/s, which a constant "
                "400 / (2 x 138) = 1.45 m/s^2 achieves. That is less than half the 3.0 m/s^2 "
                "comfort limit, so the correct response is a gentle deceleration and NOT an "
                "emergency intervention. A system that reaches emergency authority here is "
                "reading a routine deceleration as a threat."
            ),
            frames=300,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=40.0,
                initial_speed_mps=20.0,
                accel_fn=lead_brakes(2.0, start_s=1.0),
                label="lead braking at 2 m/s^2",
            ),
            expect=Expectation(
                no_collision=True,
                min_clearance_m=2.0,
                must_intervene=True,
                max_commanded_decel_mps2=3.5,
            ),
        )
    )

    scenarios.append(
        Scenario(
            name="lead_accelerates_away",
            summary="lead 25 m ahead at 20 m/s accelerates to 28 m/s",
            physics=(
                "The lead is never slower than the ego and its acceleration is positive "
                "throughout, so the true collision-avoidance requirement is identically zero "
                "for the whole run. The gap starts at 25 m, below the vehicle's own 52 m "
                "spacing policy, so a headway correction at up to the 3.0 m/s^2 comfort limit "
                "is defensible -- but only for as long as it takes the gap to open, which at a "
                "closing rate that reverses within 1 s is a couple of seconds at most. Three "
                "m/s^2 for two seconds costs 6 m/s, so a run that ever drops below 14 m/s is "
                "braking for something that is not there."
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
                max_commanded_decel_mps2=3.5,
                max_safety_state=SafetyState.LIMITED,
                min_speed_floor_mps=14.0,
                recover_within_frames=60,
            ),
        )
    )

    # ------------------------------------------------- constant range -------
    scenarios.append(
        Scenario(
            name="constant_range_motorway",
            summary="15 s of steady following at 20 m/s and 52 m, the policy spacing",
            guards="round-1 phantom AEB",
            physics=(
                "Both vehicles hold 20 m/s for 15 s. The true closing rate is identically zero "
                "and the true gap is identically 52 m, which is exactly the vehicle's own "
                "spacing policy (12 m plus a 2.0 s time gap at 20 m/s). The collision-avoidance "
                "requirement is zero, the headway requirement is zero, and there is therefore "
                "no measurement anywhere in this run that justifies a single newton of brake. "
                "Round 1 put 42 of 400 real-video frames into MIN_RISK_MANEUVER and commanded "
                "brake = 1.00 on six of them for a lead at constant range; closed loop it "
                "braked the ego from 20 m/s to 0 behind a car that never moved. The ceiling "
                "here is 1.0 m/s^2, which is the largest deceleration that could be called "
                "trim rather than intervention."
            ),
            frames=300,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=52.0,
                initial_speed_mps=20.0,
                accel_fn=lead_constant_speed(),
                label="lead keeping pace",
            ),
            expect=Expectation(
                no_collision=True,
                forbid_emergency_intervention=True,
                max_commanded_decel_mps2=1.0,
                max_safety_state=SafetyState.NOMINAL,
                min_speed_floor_mps=18.5,
                min_final_speed_mps=18.5,
            ),
        )
    )

    scenarios.append(
        Scenario(
            name="constant_range_close_follow",
            summary="steady following at 20 m/s and 20 m, a 1 s headway",
            guards="round-1 phantom AEB",
            physics=(
                "The lead holds 20 m/s, so the closing rate is identically zero and the "
                "collision-avoidance requirement is identically zero: nothing here can ever "
                "become contact without the lead first doing something, and it never does. "
                "The 20 m gap is well inside the 52 m policy spacing, so opening it at up to "
                "the 3.0 m/s^2 comfort limit is correct behaviour. Emergency authority is not: "
                "a gap that is not shrinking is not an emergency however small it is."
            ),
            frames=300,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=20.0,
                initial_speed_mps=20.0,
                accel_fn=lead_constant_speed(),
                label="lead keeping pace, close",
            ),
            expect=Expectation(
                no_collision=True,
                forbid_emergency_intervention=True,
                max_commanded_decel_mps2=3.5,
                max_safety_state=SafetyState.LIMITED,
                min_speed_floor_mps=8.0,
                recover_within_frames=100,
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
                "washes out in 4.0 m. The first few frames of a brand-new track carry no rate "
                "history, so up to five frames of over-strong response are allowed; a system "
                "that is still over-braking after a quarter of a second is not reacting, it is "
                "guessing."
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
                "counts as a hazard at all. A cut-in is not automatically an emergency, and a "
                "system that treats every new in-path track as one will brake hard several "
                "times per motorway journey."
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
                min_clearance_m=2.0,
                forbid_emergency_intervention=True,
                max_commanded_decel_mps2=3.5,
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
                "disengaged here. The 0.5 m/s^2 brake ceiling is deliberately near zero: on an "
                "empty road at a steady cruise speed the correct brake command is exactly zero."
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
                "0.8 s costs 2.8 m/s, so the ego should be at or above 12 m/s throughout and "
                "back to NOMINAL within 2 s of the dropout ending. This is the assertion that "
                "catches a degraded state that sustains itself."
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
            physics=(
                "A parked car sits 70 m ahead and is detected from frame 0; from frame 40 to "
                "frame 55 the detector returns nothing for it, then it is detected again. The "
                "car did not cease to exist because the detector blinked. At 15 m/s the ego "
                "covers 12 m during the miss, and full-authority braking from 15 m/s needs "
                "15.0 m plus 2.0 m of clearance, so even a system that coasted through the "
                "entire dropout and only reacted on re-acquisition has ample room. Contact "
                "here means the dropout was allowed to release an intervention that the world "
                "had not stopped requiring."
            ),
            frames=250,
            ego_speed_mps=15.0,
            lead=LeadSpec(
                initial_gap_m=70.0,
                initial_speed_mps=0.0,
                accel_fn=lead_stationary(),
                label="parked car",
            ),
            perception=PerceptionSpec(miss_frames=tuple(range(40, 56))),
            expect=Expectation(
                no_collision=True,
                min_clearance_m=0.5,
                must_intervene=True,
            ),
        )
    )

    # ----------------------------------------------------- track identity ---
    scenarios.append(
        Scenario(
            name="reid_during_steady_follow",
            summary="steady follow at 20 m/s and 52 m; the tracker re-ids the lead three times",
            guards="round-1 phantom AEB (new-track rate seed)",
            physics=(
                "The lead holds 20 m/s at a constant 52 m for the whole run: the true closing "
                "rate is identically zero. On frames 60, 120 and 180 the tracker assigns a new "
                "id to the same physical vehicle. A track id is bookkeeping; it carries no "
                "information about motion. Any estimator that seeds a new track's closing rate "
                "at the ego speed will read -20 m/s of closure at 52 m and demand an emergency "
                "brake, which is precisely the phantom found at ego 15 m/s for every new "
                "in-path track inside about 90 m. Nothing physical changed at frame 60, so "
                "nothing in the command may change either."
            ),
            frames=300,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=52.0,
                initial_speed_mps=20.0,
                accel_fn=lead_constant_speed(),
                label="lead keeping pace",
            ),
            perception=PerceptionSpec(reid_frames=(60, 120, 180)),
            expect=Expectation(
                no_collision=True,
                forbid_emergency_intervention=True,
                max_commanded_decel_mps2=1.0,
                max_safety_state=SafetyState.NOMINAL,
                min_speed_floor_mps=18.5,
                min_final_speed_mps=18.5,
            ),
        )
    )

    scenarios.append(
        Scenario(
            name="range_jump_during_steady_follow",
            summary="steady follow at 20 m/s and 52 m; the reported range steps 10 m closer",
            guards="round-1 phantom AEB (range re-seed)",
            physics=(
                "The true gap is 52 m for the whole run and the true closing rate is zero. At "
                "frame 60 the reported range steps from 52 m to 42 m in one 50 ms frame and "
                "stays there. Differentiating that step gives 200 m/s of closure, which no "
                "road vehicle can produce; the only physical reading of a 10 m step in 50 ms "
                "is that the measurement changed, not the world. The system may correct its "
                "headway towards the new apparent 42 m at up to comfort, and must return to "
                "NOMINAL once the apparent range has been steady for a few seconds. It may not "
                "brake as though 200 m/s of closure were real."
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
                max_commanded_decel_mps2=3.5,
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
            summary="a mock lane centre 460 px off while a parked car sits 65 m dead ahead",
            guards="round-1 lane-error blindness",
            physics=(
                "The lane estimate is mock, its confidence is zero and it places the lane "
                "centre 460 px left of the image centre. The obstacle is real, stationary, "
                "dead ahead at 65 m, and the ego closes on it at 20 m/s. Full-authority "
                "braking needs 26.3 m plus 2.0 m of clearance, so there is 36 m of margin. A "
                "lane estimate may widen the corridor a hazard is searched in -- more lane "
                "means more places a threat can be -- but it must never move or narrow it, "
                "because a lane detector that fails towards the kerb would then be able to "
                "hide a car that is directly in front of the bumper. A collision here is a "
                "lateral fault causing a longitudinal one."
            ),
            frames=250,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=65.0,
                initial_speed_mps=0.0,
                accel_fn=lead_stationary(),
                label="parked car",
            ),
            perception=PerceptionSpec(
                lane_center_bias_px=-460.0, lane_is_mock=True, lane_confidence=0.0
            ),
            expect=Expectation(
                no_collision=True,
                min_clearance_m=0.5,
                must_intervene=True,
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
                "longitudinal hazard of any kind, and the state must stay NOMINAL for the whole "
                "15 s. The ego must also stay inside its lane: 1.75 m either side of the centre "
                "is half of a 3.5 m lane."
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
                "the 1.75 m half-lane. So a straight-ahead hold is survivable here and a held "
                "or oscillating steering command is not. Once perception returns the road is "
                "empty and every measurement is benign, so the system must be back in NOMINAL "
                "within 3 s and stay there."
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
                "which from 20 m/s takes 5.7 s and 57 m; the run gives it 9 s. Continuing to "
                "cruise blind fails, and so does latching DISENGAGE, because disengaging hands "
                "a moving vehicle to nobody -- there is no driver in this loop to catch it. "
                "The 5.0 m/s^2 ceiling separates a controlled stop from an emergency stop: "
                "nothing was seen, so nothing warrants full authority."
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
                "The lead brakes at 6 m/s^2 from t=0.5 s at a 20 m gap, so at t=2.5 s the ego "
                "is mid-intervention with the brake applied. The run then ends -- a clean exit, "
                "a bounded frame count, a SIGTERM; the reason does not matter. Leaving a loop "
                "is not a command: whatever was last written to the actuators stays latched "
                "there. So the exit must WRITE something, that something must have zero "
                "throttle, and it must not release a brake that the previous frame was "
                "applying. Emitting nothing, or emitting a neutral command, releases the brake "
                "of a vehicle that is still closing on a decelerating car."
            ),
            frames=50,
            ego_speed_mps=20.0,
            lead=LeadSpec(
                initial_gap_m=20.0,
                initial_speed_mps=20.0,
                accel_fn=lead_brakes(6.0, start_s=0.5),
                label="lead braking at 6 m/s^2",
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
                "of a clear road and the system must be in NOMINAL within 2 s. All three of "
                "the self-sustaining loops in this module's history -- steering rate feeding "
                "plan acceleration feeding lane departure -- fail exactly here, and nowhere "
                "else in this library, because this is the only case that asks the system to "
                "let go."
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
