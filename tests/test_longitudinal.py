"""Property tests for the longitudinal (ACC + AEB) law.

These are sweeps, not point assertions. The defects this file exists to catch --
a non-monotone speed map, a discontinuity the range noise can chatter across, an
unbounded target-speed step, a perception dropout read as "road clear" -- are all
invisible to a test that only checks ``0 < v < cruise``.

Every randomised test uses a FIXED seed so a failure is reproducible.
"""

from __future__ import annotations

import logging
import math
import random

import pytest

from adas.core.exceptions import ValidationError
from adas.planning.longitudinal import (
    LeadVehicle,
    LogGate,
    LongitudinalLimits,
    LongitudinalPlanner,
)

SEED = 20240913
DT = 0.05

EGO_SPEEDS = (0.0, 2.5, 7.0, 12.0, 15.0)
RANGE_RATES = (-20.0, -12.0, -6.0, -2.0, -0.5, 0.0, 3.0, 8.0)


def _planner(**kwargs) -> LongitudinalPlanner:
    return LongitudinalPlanner(LongitudinalLimits(**kwargs))


def _raw(planner: LongitudinalPlanner, distance_m: float, rate: float, ego: float) -> float:
    lead = LeadVehicle(distance_m=distance_m, range_rate_mps=rate)
    return planner.raw_target_speed_mps(lead, ego)[0]


# --------------------------------------------------------------------------- #
# Monotonicity -- the ADAS-DEC-02 blocker
# --------------------------------------------------------------------------- #


def test_target_speed_non_decreasing_in_distance():
    """Sweep the whole domain: closing the gap must never raise the target speed.

    The previous law stepped 6.00 -> 14.875 m/s as the range crossed 12.0 m.
    """
    planner = _planner()
    for ego in EGO_SPEEDS:
        for rate in RANGE_RATES:
            previous = -1.0
            for step in range(0, 1001):
                distance = step * 0.1  # 0 .. 100 m
                value = _raw(planner, distance, rate, ego)
                assert value >= previous - 1e-9, (
                    "target speed fell as the gap grew: d=%.2f m, v_ego=%.1f, "
                    "v_rel=%.1f -> %.4f after %.4f" % (distance, ego, rate, value, previous)
                )
                previous = value


def test_target_speed_non_increasing_in_closing_rate():
    """Sweep closing rate: closing faster must never raise the target speed."""
    planner = _planner()
    for ego in EGO_SPEEDS:
        for step_d in range(0, 41):
            distance = 2.0 + step_d * 2.0  # 2 .. 82 m
            previous = float("inf")
            for step_r in range(0, 121):
                closing = step_r * 0.25  # 0 .. 30 m/s of closing
                value = _raw(planner, distance, -closing, ego)
                assert value <= previous + 1e-9, (
                    "target speed rose as closing rate grew: d=%.1f m, v_ego=%.1f, "
                    "closing=%.2f -> %.4f after %.4f" % (distance, ego, closing, value, previous)
                )
                previous = value


def test_comfort_law_is_lipschitz_in_distance():
    """The comfort law has no cliff: dv/dd is exactly k_distance."""
    planner = _planner()
    k = planner.limits.k_distance
    for ego in EGO_SPEEDS:
        for rate in RANGE_RATES:
            for step in range(0, 1000):
                d0 = step * 0.1
                d1 = d0 + 0.1
                v0 = planner.equilibrium_speed_mps(d0, rate, ego)
                v1 = planner.equilibrium_speed_mps(d1, rate, ego)
                assert abs(v1 - v0) <= k * 0.1 + 1e-9


def test_no_discontinuity_at_the_old_min_follow_threshold():
    """The 12.0 m branch boundary of the old law is now smooth."""
    planner = _planner()
    ego = 15.0
    below = planner.equilibrium_speed_mps(11.9, 0.0, ego)
    at = planner.equilibrium_speed_mps(12.0, 0.0, ego)
    above = planner.equilibrium_speed_mps(12.1, 0.0, ego)
    assert abs(at - below) < 0.05
    assert abs(above - at) < 0.05


# --------------------------------------------------------------------------- #
# Spacing policy
# --------------------------------------------------------------------------- #


def test_equilibrium_is_the_constant_time_gap_fixed_point():
    """At d = d0 + T*v with zero relative speed, the law asks for exactly v."""
    planner = _planner()
    lim = planner.limits
    for step in range(0, 31):
        speed = step * 0.5
        if speed > lim.cruise_speed_mps:
            break
        distance = lim.min_follow_distance_m + lim.time_gap_s * speed
        assert planner.equilibrium_speed_mps(distance, 0.0, speed) == pytest.approx(speed, abs=1e-9)


def test_steady_state_gap_respects_the_time_gap():
    """Closed-loop: settle the target and check the resulting headway."""
    planner = _planner()
    lim = planner.limits
    speed = 10.0
    distance = lim.min_follow_distance_m + lim.time_gap_s * speed
    for _ in range(200):
        decision = planner.plan(
            LeadVehicle(distance_m=distance, range_rate_mps=0.0), speed, True, DT
        )
        speed = decision.target_speed_mps
    assert distance / max(speed, 1e-6) >= lim.time_gap_s


def test_closer_than_the_desired_gap_commands_less_than_ego_speed():
    """The 'too close' regime must be slower than cruising, not faster.

    The old law's ``d < min_follow`` branch implemented a 0.8 s gap, i.e. it was
    2.5x MORE aggressive than the nominal branch.
    """
    planner = _planner()
    ego = 15.0
    for step in range(1, 60):
        distance = step * 0.5  # 0.5 .. 29.5 m, all inside the 42 m desired gap
        assert _raw(planner, distance, 0.0, ego) < ego


# --------------------------------------------------------------------------- #
# AEB stage
# --------------------------------------------------------------------------- #


def test_aeb_fires_for_a_stopped_lead_at_short_range():
    planner = _planner()
    lead = LeadVehicle(distance_m=10.0, range_rate_mps=-15.0)
    target, reason, aeb, ttc, required = planner.raw_target_speed_mps(lead, 15.0)
    assert aeb is True
    assert target == 0.0
    assert "aeb" in reason
    assert ttc < planner.limits.aeb_ttc_s or required > planner.limits.aeb_required_decel_mps2


def test_aeb_does_not_fire_when_the_gap_is_opening():
    planner = _planner()
    lead = LeadVehicle(distance_m=8.0, range_rate_mps=+5.0)
    _, _, aeb, ttc, required = planner.raw_target_speed_mps(lead, 10.0)
    assert aeb is False
    assert math.isinf(ttc)
    assert required == 0.0


def test_hazard_maths_matches_the_closed_form():
    planner = _planner(standstill_gap_m=4.0)
    ttc, required = planner.hazard(24.0, -10.0)
    assert ttc == pytest.approx(20.0 / 10.0)
    assert required == pytest.approx(100.0 / (2.0 * 20.0))


def test_inside_the_standstill_gap_target_is_zero():
    planner = _planner()
    assert _raw(planner, 3.9, 0.0, 5.0) == 0.0
    assert _raw(planner, 0.0, 0.0, 5.0) == 0.0


# --------------------------------------------------------------------------- #
# Rate limiting -- ADAS-DEC-11 / ADAS-DEC-19
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Fail-safe direction -- the ADAS-DEC-04 blocker
# --------------------------------------------------------------------------- #


def test_perception_dropout_never_returns_cruise():
    """A blind camera must not be read as an empty road."""
    planner = _planner()
    ego = 12.0
    planner.plan(LeadVehicle(distance_m=40.0, range_rate_mps=0.0), ego, True, DT)
    previous = planner.previous_target_mps
    for index in range(1, 121):
        decision = planner.plan(None, ego, perception_valid=False, dt_s=DT)
        assert decision.degraded is True
        assert decision.dropout_frames == index
        assert decision.target_speed_mps <= previous + 1e-9, "target rose during a dropout"
        assert decision.target_speed_mps < planner.limits.cruise_speed_mps
        previous = decision.target_speed_mps
    assert previous == pytest.approx(0.0, abs=1e-9), "dropout must end in a controlled stop"


def test_unknown_ego_speed_degrades_and_never_guesses():
    planner = _planner()
    decision = planner.plan(LeadVehicle(distance_m=40.0, range_rate_mps=0.0), None, True, DT)
    assert decision.degraded is True
    assert decision.reason.startswith("ego_speed_unavailable")
    assert decision.target_speed_mps == 0.0
    for bad in (float("nan"), float("inf"), -1.0):
        planner.reset()
        assert planner.plan(None, bad, True, DT).reason.startswith("ego_speed_unavailable")


def test_non_finite_lead_range_is_a_fault_not_an_empty_road():
    planner = _planner()
    for bad in (float("nan"), float("inf"), -3.0):
        planner.reset()
        target, reason, aeb, _, _ = planner.raw_target_speed_mps(
            LeadVehicle(distance_m=bad, range_rate_mps=0.0), 15.0
        )
        assert target == 0.0
        assert aeb is True
        assert reason == "invalid_range"


def test_coasting_lead_blocks_acceleration():
    """A range nobody measured this frame must not release the throttle."""
    planner = _planner()
    planner.plan(LeadVehicle(distance_m=40.0, range_rate_mps=0.0), 5.0, True, DT)
    baseline = planner.previous_target_mps
    decision = planner.plan(
        LeadVehicle(distance_m=40.0, range_rate_mps=0.0, frames_since_measurement=2), 5.0, True, DT
    )
    assert decision.target_speed_mps <= baseline + 1e-9
    assert "coast" in decision.reason


# --------------------------------------------------------------------------- #
# Envelope and configuration
# --------------------------------------------------------------------------- #


def test_target_is_always_inside_the_envelope():
    rng = random.Random(SEED + 2)
    planner = _planner()
    for _ in range(5000):
        lead = None
        if rng.random() < 0.8:
            lead = LeadVehicle(
                distance_m=rng.uniform(0.0, 120.0), range_rate_mps=rng.uniform(-30.0, 15.0)
            )
        ego = rng.uniform(0.0, 20.0)
        decision = planner.plan(lead, ego, rng.random() > 0.1, rng.uniform(0.02, 0.2))
        assert math.isfinite(decision.target_speed_mps)
        assert 0.0 <= decision.target_speed_mps <= planner.limits.cruise_speed_mps


def test_reset_clears_the_rate_limiter():
    planner = _planner()
    planner.plan(LeadVehicle(distance_m=60.0, range_rate_mps=0.0), 15.0, True, DT)
    assert planner.previous_target_mps is not None
    planner.reset()
    assert planner.previous_target_mps is None
    assert planner.dropout_frames == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cruise_speed_mps": 0.0},
        {"time_gap_s": 0.0},
        {"k_distance": 0.0},
        {"k_speed": -1.0},
        {"standstill_gap_m": 20.0},
        {"emergency_decel_mps2": 1.0},
        {"aeb_ttc_s": 2.0, "warn_ttc_s": 1.0},
        {"mrm_after_dropouts": 0},
    ],
)
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        LongitudinalLimits(**kwargs)


# --------------------------------------------------------------------------- #
# Log volume -- a latched condition must not be a per-frame WARNING
# --------------------------------------------------------------------------- #


def test_log_gate_emits_on_entry_then_suppresses_then_on_exit():
    gate = LogGate(period_s=10.0)
    assert gate.mark(now_s=0.0) == (True, 0)
    assert gate.active is True
    for index in range(1, 100):
        assert gate.mark(now_s=index * 0.05) == (False, 0)
    assert gate.suppressed == 99
    # Still inside the period at t = 4.95 s; the next mark past 10 s repeats.
    assert gate.mark(now_s=10.0) == (True, 99)
    assert gate.suppressed == 0
    assert gate.clear() == (True, 0)
    assert gate.active is False
    assert gate.clear() == (False, 0), "exit must be logged exactly once"


def test_log_gate_period_zero_never_repeats():
    gate = LogGate(period_s=0.0)
    assert gate.mark(now_s=0.0) == (True, 0)
    for index in range(1000):
        assert gate.mark(now_s=float(index)) == (False, 0)


def test_log_gate_reset_forgets_the_latch():
    gate = LogGate(period_s=10.0)
    gate.mark(now_s=0.0)
    gate.mark(now_s=0.05)
    gate.reset()
    assert gate.active is False
    assert gate.suppressed == 0
    assert gate.mark(now_s=0.10) == (True, 0)


def test_missing_ego_speed_logs_once_not_once_per_frame(caplog):
    planner = _planner()
    with caplog.at_level(logging.WARNING, logger="adas.planning.longitudinal"):
        for _ in range(600):
            decision = planner.plan(None, None, True, DT)
    assert decision.reason.startswith("ego_speed_unavailable")
    lines = [r for r in caplog.records if "no usable ego speed" in r.getMessage()]
    assert len(lines) == 1, "%d warnings for one latched condition" % len(lines)


def test_ego_speed_recovery_is_logged_once_and_re_arms_the_gate(caplog):
    planner = _planner()
    with caplog.at_level(logging.DEBUG, logger="adas.planning.longitudinal"):
        for _ in range(200):
            planner.plan(None, None, True, DT)
        for _ in range(200):
            planner.plan(None, 12.0, True, DT)
            assert planner.ego_speed_available is True
        for _ in range(200):
            planner.plan(None, None, True, DT)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    infos = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "ego speed restored" in r.getMessage()
    ]
    assert len(warnings) == 2, "one WARNING per entry into the condition, got %d" % len(warnings)
    assert len(infos) == 1, "recovery must be logged exactly once"


def test_reset_re_arms_the_ego_speed_log_gate(caplog):
    planner = _planner()
    with caplog.at_level(logging.DEBUG, logger="adas.planning.longitudinal"):
        for _ in range(50):
            planner.plan(None, None, True, DT)
        planner.reset()
        assert planner.ego_speed_available is True
        for _ in range(50):
            planner.plan(None, None, True, DT)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 2


# --------------------------------------------------------------------------- #
# Range-rate cross-check
#
# Regression for the second half of the AEB blocker: with a stationary lead 25 m
# ahead at 15 m/s the tracker's velocity channel reported 0, so every AEB
# predicate read "not closing", the planner said follow_gap on all 40 frames and
# the controller commanded brake 0.000 on all 40. The planner now measures the
# range derivative itself and only overrides a rate that its own measurement
# contradicts -- see the module docstring for why that is not the same thing as
# fabricating a closing rate out of the ego speed.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"range_rate_cross_check_window": 4},
        {"range_rate_disagreement_mps": 0.0},
        {"range_rate_disagreement_mps": float("nan")},
        {"range_rate_significance_sigma": 0.0},
        {"range_rate_significance_sigma": float("inf")},
    ],
)


# --------------------------------------------------------------------------- #
# The evidence-gated law: what the planner publishes, and what it refuses to
# --------------------------------------------------------------------------- #
#
# The planner's braking output is ``SpeedDecision.decel_demand_mps2``, not a
# target speed of zero.  A target speed cannot express a deceleration: the
# planner's rate-limited target falls 0.15 m/s per frame for a 3 m/s^2 request,
# a proportional speed law needs a 20 m/s error to answer that, and the brake
# the primary path actually produced while trailing a comfort ramp was an eighth
# of what was asked for -- which left the safety arbiter as the only component
# in the vehicle that really braked.


def _feed(planner, gap0, closing, ego, frames, dt=DT, track_id=1, noise=None):
    """Drive the planner down a closing approach and return every decision.

    The range is what a camera reports; the planner differentiates it itself, so
    the reported ``range_rate_mps`` is deliberately left at zero throughout --
    nothing here may depend on the tracker's rate channel.
    """
    out = []
    gap = gap0
    for index in range(frames):
        reported = gap if noise is None else gap + noise(index)
        lead = LeadVehicle(
            distance_m=max(0.1, reported),
            range_rate_mps=0.0,
            track_id=track_id,
            capture_token=index + 1,
        )
        out.append(planner.plan(lead, ego, True, dt))
        gap -= closing * dt
    return out


def test_no_braking_at_all_until_a_closing_rate_has_been_measured():
    """The founding defect: acting on the ``-ego_speed`` stationary prior.

    A track the planner has never seen carries no rate.  The only prior available
    is "assume it is stationary in the world", which for a lead holding station
    fabricates the ego's whole speed as a closure and brakes the vehicle to a
    standstill behind a car that never moved.  There is no such prior here: until
    the window carries enough distinct captures the avoidance law contributes
    nothing at all, and the fallback is the time-gap law.
    """
    planner = _planner()
    first = planner.plan(
        LeadVehicle(distance_m=15.0, range_rate_mps=-15.0, track_id=1, capture_token=1),
        15.0, True, DT,
    )
    assert first.rate_is_measured is False
    assert first.required_decel_mps2 == 0.0, "an avoidance requirement out of no measurement"
    assert first.aeb_active is False
    # What IS allowed on an unmeasured frame is the time-gap law, which rests on
    # the measured RANGE alone and is bounded at the headway allowance.
    assert first.decel_demand_mps2 <= planner.limits.headway_decel_mps2 + 1e-9


def test_a_measured_closure_produces_a_deceleration_demand_not_a_zero_target():
    planner = _planner()
    decisions = _feed(planner, gap0=15.0, closing=15.0, ego=15.0, frames=8)
    measured = [d for d in decisions if d.rate_is_measured]
    assert measured, "the planner never measured the closure it was shown"
    assert measured[-1].decel_demand_mps2 > 0.0
    assert measured[-1].required_decel_mps2 > 3.0, "a 15 m/s closure at 15 m is an emergency"
    # And the demand is a DECELERATION: the target speed is a comfort request and
    # is not the channel the braking travels down.
    assert measured[-1].decel_demand_mps2 == pytest.approx(planner.demand_mps2)


def test_a_matched_speed_lead_never_earns_more_than_the_headway_allowance():
    """The constant-range phantom, in the planner.

    Twelve metres is well inside the 42 m policy gap at 15 m/s, so OPENING the
    gap is correct and the planner does it -- but opening a gap is headway
    keeping, and headway keeping is bounded at ``headway_decel_mps2``.  The
    failure this pins is the one that took a 20 m/s ego to a standstill behind a
    car holding a constant 32.5 m.
    """
    planner = _planner()
    decisions = _feed(planner, gap0=12.0, closing=0.0, ego=15.0, frames=200)
    peak = max(d.decel_demand_mps2 for d in decisions)
    assert peak <= planner.limits.headway_decel_mps2 + 1e-9, (
        "matched-speed follow demanded %.2f m/s^2" % peak
    )
    assert all(not d.aeb_active for d in decisions)


def test_range_noise_alone_never_earns_collision_avoidance_authority():
    """+/-0.30 m of range noise on a lead that is not closing.

    The historic false positive.  A least-squares slope over a short window has a
    standard error of metres per second at this noise level, so an estimator that
    believed every sample would see several m/s of spurious closure; the four
    sigma bound is what stops it reaching the braking law.
    """
    rng = random.Random(SEED + 7)
    planner = _planner()
    decisions = _feed(
        planner, gap0=20.0, closing=0.0, ego=20.0, frames=400,
        noise=lambda i: rng.gauss(0.0, 0.30),
    )
    peak = max(d.decel_demand_mps2 for d in decisions)
    assert peak <= planner.limits.headway_decel_mps2 + 1e-9, (
        "range noise manufactured %.2f m/s^2 of braking" % peak
    )


def test_range_noise_does_not_suppress_a_real_closure_either():
    """The mirror: a system can be made noise-proof by ignoring the sensor."""
    rng = random.Random(SEED + 8)
    planner = _planner()
    decisions = _feed(
        planner, gap0=36.0, closing=20.0, ego=20.0, frames=30,
        noise=lambda i: rng.gauss(0.0, 0.30),
    )
    assert max(d.decel_demand_mps2 for d in decisions) >= 3.5, (
        "a stopped obstacle at 36 m and 20 m/s was not braked for"
    )
    fired = next(i for i, d in enumerate(decisions) if d.decel_demand_mps2 > 0.0)
    assert fired <= 5, "took %d frames to react to an unmissable closure" % fired


def test_the_demand_is_jerk_shaped_at_the_specifications_own_ceilings():
    planner = _planner()
    lim = planner.limits
    decisions = _feed(planner, gap0=36.0, closing=20.0, ego=20.0, frames=60)
    previous = 0.0
    for decision in decisions:
        rise = decision.decel_demand_mps2 - previous
        # The band comes from where the demand is HEADING, exactly as the
        # specification's own ceiling does: a ramp bound for emergency grade is a
        # collision-avoidance action for the whole of its rise, because a ramp
        # that had to pause at the comfort limit on its way to 8 m/s^2 would not
        # be one.
        heading = max(decision.required_decel_mps2, decision.decel_demand_mps2)
        ceiling = (
            lim.emergency_jerk_mps3
            if heading >= lim.emergency_grade_mps2
            else lim.comfort_jerk_mps3
        )
        assert rise <= ceiling * DT + 1e-9, "demand rose at %.1f m/s^3" % (rise / DT)
        previous = decision.decel_demand_mps2


def test_a_range_jump_is_a_re_anchor_not_two_hundred_metres_per_second():
    """A 10 m step in one 50 ms frame implies 200 m/s of closure."""
    planner = _planner()
    for index in range(12):
        planner.plan(
            LeadVehicle(distance_m=52.0, range_rate_mps=0.0, track_id=1, capture_token=index + 1),
            20.0, True, DT,
        )
    after = []
    for index in range(12, 40):
        after.append(planner.plan(
            LeadVehicle(distance_m=42.0, range_rate_mps=0.0, track_id=1, capture_token=index + 1),
            20.0, True, DT,
        ))
    assert max(d.decel_demand_mps2 for d in after) <= planner.limits.headway_decel_mps2 + 1e-9


def test_a_detection_miss_holds_the_demand_rather_than_dropping_it():
    """A detector that produced nothing has not reported an empty road."""
    planner = _planner()
    decisions = _feed(planner, gap0=30.0, closing=20.0, ego=20.0, frames=12)
    held = decisions[-1].decel_demand_mps2
    assert held > 0.0, "precondition: the planner was braking"
    missed = planner.plan(None, 20.0, True, DT)
    assert missed.decel_demand_mps2 == pytest.approx(held, abs=1e-9)
    for _ in range(planner.limits.miss_hold_frames + 40):
        last = planner.plan(None, 20.0, True, DT)
    assert last.decel_demand_mps2 == 0.0, "the hold never expired"


def test_a_perception_dropout_holds_then_makes_a_controlled_stop():
    planner = _planner()
    lim = planner.limits
    planner.plan(
        LeadVehicle(distance_m=80.0, range_rate_mps=0.0, track_id=1, capture_token=1),
        15.0, True, DT,
    )
    for index in range(1, lim.blind_hold_frames + 1):
        decision = planner.plan(None, 15.0, perception_valid=False, dt_s=DT)
        assert decision.degraded is True
        assert decision.decel_demand_mps2 == 0.0, "a single blink is not a reason to brake"
    for _ in range(200):
        decision = planner.plan(None, 15.0, perception_valid=False, dt_s=DT)
    assert decision.decel_demand_mps2 == pytest.approx(lim.mrm_decel_mps2, abs=1e-6)
    assert decision.decel_demand_mps2 < lim.emergency_grade_mps2, (
        "a minimum-risk stop is a controlled stop, not an emergency one"
    )


def test_reset_clears_the_evidence_and_the_shaper():
    planner = _planner()
    _feed(planner, gap0=30.0, closing=20.0, ego=20.0, frames=12)
    assert planner.demand_mps2 > 0.0
    planner.reset()
    assert planner.demand_mps2 == 0.0
    first = planner.plan(
        LeadVehicle(distance_m=30.0, range_rate_mps=0.0, track_id=1, capture_token=1),
        20.0, True, DT,
    )
    assert first.rate_is_measured is False, "the window survived a reset"


# --------------------------------------------------------------------------- #
# Property-level bounds on the published target speed
#
# These four were dropped when the cross-check mechanism they were written
# against was removed.  Three of the four never mentioned that mechanism: they
# bound the PUBLISHED TARGET SPEED, which is still published, and the properties
# they assert -- no chatter across the avoidance boundary, a bounded step under
# adversarial input, no acceleration when a lead flickers, no jump back to cruise
# after a dropout -- are exactly the ones a rewritten law is most likely to lose.
# Restored, and re-pointed at the current API.
# --------------------------------------------------------------------------- #


def test_target_speed_rate_is_bounded_over_an_adversarial_sequence():
    """Randomised ranges must not produce an unbounded target-speed step.

    Four thousand frames of uniformly random range and range rate, which is a
    harder input than any road: the gap teleports between 0 and 60 m every 50 ms.
    The published target must still rise no faster than ``max_accel_mps2`` and
    fall no faster than ``max_decel_mps2``, except on an AEB frame, where it is
    published at 0 m/s immediately and deliberately -- a target that trailed the
    vehicle down is how the primary path came to contribute 0.06 m/s^2 during an
    emergency stop.
    """
    rng = random.Random(SEED)
    planner = _planner()
    lim = planner.limits
    ego = 15.0
    previous = None
    saw_aeb = False
    for _ in range(4000):
        distance = rng.uniform(0.0, 60.0)
        rate = rng.uniform(-20.0, 5.0)
        decision = planner.plan(
            LeadVehicle(distance_m=distance, range_rate_mps=rate), ego, True, DT
        )
        if previous is not None:
            delta = decision.target_speed_mps - previous
            assert delta <= lim.max_accel_mps2 * DT + 1e-9, "target accelerated too fast"
            if decision.aeb_active:
                saw_aeb = True
                assert decision.target_speed_mps == 0.0, "an AEB frame must publish 0 m/s"
            else:
                assert -delta <= lim.max_decel_mps2 * DT + 1e-9, "target decelerated too fast"
        previous = decision.target_speed_mps
        ego = max(0.0, min(lim.cruise_speed_mps, decision.target_speed_mps))
    # ...and it never reached the AEB branch at all, which is the SECOND half of
    # the property and the one the current law adds. The old law read the
    # reported ``range_rate_mps`` on the frame it arrived, so a single uniform
    # draw of -20 m/s was an emergency; this one fits a rate to its own window of
    # RAW ranges and rejects a step beyond ``evidence.jump_m`` as a re-anchor, so
    # a range that teleports 60 m every 50 ms has NO measurable closure and earns
    # no avoidance authority whatsoever. A sequence with no coherent motion in it
    # must produce no emergency; the coherent case is the test below.
    assert not saw_aeb, (
        "a range teleporting uniformly over 0-60 m produced an emergency; the "
        "rate came from somewhere other than a measurement"
    )


def test_a_coherent_closing_approach_publishes_a_zero_target_without_rate_limiting():
    """The AEB exemption from the rate limit, on the only input that can reach it.

    The pair of the adversarial test above: a physically coherent 15 m/s approach
    to a stopped car, so the closure IS measurable and the avoidance law does earn
    its authority. On the frame the demand reaches emergency grade the published
    target must step to 0 m/s in one frame rather than trailing the vehicle down
    at the comfort rate.
    """
    planner = _planner()
    ego = 15.0
    distance = 45.0
    saw_aeb = False
    for frame in range(200):
        decision = planner.plan(
            LeadVehicle(distance_m=distance, range_rate_mps=-ego, capture_token=frame + 1),
            ego,
            True,
            DT,
        )
        if decision.aeb_active:
            saw_aeb = True
            assert decision.target_speed_mps == 0.0, decision
        distance = max(0.5, distance - ego * DT)
        ego = max(0.0, ego - decision.decel_demand_mps2 * DT)
        if ego <= 0.01:
            break
    assert saw_aeb, "a 15 m/s approach to a stopped car never reached emergency grade"


def test_dithering_range_across_the_aeb_boundary_does_not_chatter():
    """One pixel of box-height noise used to swing the command 6.0 <-> 14.9 m/s.

    A thousand frames whose range dithers +/-0.6 m about the range at which the
    required deceleration crosses the emergency grade. A discontinuity there is
    invisible to a monotonicity test -- both branches are monotone -- and it is
    what a box-height estimator's own noise rides on.
    """
    rng = random.Random(SEED + 1)
    planner = _planner()
    lim = planner.limits
    previous = None
    for _ in range(1000):
        distance = 26.5 + rng.uniform(-0.6, 0.6)
        decision = planner.plan(
            LeadVehicle(distance_m=distance, range_rate_mps=-15.0), 15.0, True, DT
        )
        if previous is not None:
            assert abs(decision.target_speed_mps - previous) <= (
                max(lim.max_accel_mps2, lim.emergency_decel_mps2) * DT + 1e-9
            )
        previous = decision.target_speed_mps


def test_comfort_decel_bounds_the_target_when_a_lead_disappears_and_reappears():
    """A lead flickering in and out on alternate frames must not free the throttle.

    Two hundred frames alternating between "car 8 m ahead" and "no car". Each
    reappearance re-establishes the same close lead, so nothing about the world
    ever justified accelerating; a law that treats the empty frames as a clear
    road ratchets the target up by ``max_accel_mps2 * dt`` every other frame.
    """
    planner = _planner()
    lim = planner.limits
    ego = 5.0
    previous = None
    for index in range(200):
        lead = None if index % 2 == 0 else LeadVehicle(distance_m=8.0, range_rate_mps=0.0)
        decision = planner.plan(lead, ego, True, DT)
        if previous is not None:
            assert decision.target_speed_mps - previous <= lim.max_accel_mps2 * DT + 1e-9
        previous = decision.target_speed_mps


def test_dropout_recovery_resumes_from_the_ramped_value_not_from_cruise():
    """Perception coming back is not permission to jump to cruise.

    Five blind frames ramp the target down. The frame perception recovers must
    continue from the ramped value, bounded by one frame of ``max_accel_mps2`` --
    not from ``cruise_speed_mps``, which would put a step of several m/s through
    the controller on the first frame after every dropout.
    """
    planner = _planner()
    planner.plan(None, 15.0, perception_valid=True, dt_s=DT)
    for _ in range(5):
        planner.plan(None, 15.0, perception_valid=False, dt_s=DT)
    ramped = planner.previous_target_mps
    decision = planner.plan(None, 15.0, perception_valid=True, dt_s=DT)
    assert decision.target_speed_mps <= ramped + planner.limits.max_accel_mps2 * DT + 1e-9
