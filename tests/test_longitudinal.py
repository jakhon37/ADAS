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


def test_target_speed_rate_is_bounded_over_an_adversarial_sequence():
    """Randomised ranges must not produce an unbounded target-speed step.

    The rate limit is DELIBERATELY one-sided with respect to AEB: an AEB frame
    publishes 0 m/s immediately (see
    ``test_aeb_target_is_published_immediately_not_rate_limited``), everything
    else is bounded by ``max_decel_mps2`` down and ``max_accel_mps2`` up.
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
        decision = planner.plan(LeadVehicle(distance_m=distance, range_rate_mps=rate), ego, True, DT)
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
    assert saw_aeb, "sweep never exercised the AEB branch; the test proves nothing"


def test_aeb_target_is_published_immediately_not_rate_limited():
    """Regression: the AEB target must LEAD the vehicle, not trail it.

    The planner used to rate-limit its own AEB target at
    ``emergency_decel_mps2 * dt`` (0.4 m/s per 50 ms frame).  The published target
    therefore tracked the measured speed down, the residual speed error stayed near
    0.4 m/s, and a proportional controller downstream commanded essentially no
    brake -- every metre of real braking came from the safety arbiter instead.
    """
    planner = _planner()
    lim = planner.limits
    # Cruise first so the rate limiter is primed at cruise speed.
    planner.plan(LeadVehicle(distance_m=80.0, range_rate_mps=0.0), 15.0, True, DT)
    assert planner.previous_target_mps == pytest.approx(15.0, abs=1e-9)

    # A cut-in at 15 m closing at 15 m/s: ttc = (15 - 4) / 15 = 0.73 s < 0.9 s.
    decision = planner.plan(LeadVehicle(distance_m=15.0, range_rate_mps=-15.0), 15.0, True, DT)
    assert decision.aeb_active is True
    assert decision.reason.startswith("aeb_")
    assert decision.target_speed_mps == 0.0, (
        "AEB published %.3f m/s instead of 0; the old rate limiter would have "
        "published %.3f" % (
            decision.target_speed_mps, 15.0 - lim.emergency_decel_mps2 * DT
        )
    )
    # The speed error the controller sees is the whole speed, not 0.4 m/s.
    assert 15.0 - decision.target_speed_mps == pytest.approx(15.0, abs=1e-9)


def test_aeb_recovery_is_still_rate_limited_upward():
    """Leaving AEB must not step the target back up."""
    planner = _planner()
    lim = planner.limits
    planner.plan(LeadVehicle(distance_m=15.0, range_rate_mps=-15.0), 15.0, True, DT)
    assert planner.previous_target_mps == 0.0
    previous = 0.0
    for _ in range(50):
        decision = planner.plan(LeadVehicle(distance_m=90.0, range_rate_mps=0.0), 15.0, True, DT)
        assert decision.aeb_active is False
        assert decision.target_speed_mps - previous <= lim.max_accel_mps2 * DT + 1e-9
        previous = decision.target_speed_mps
    assert 0.0 < previous < lim.cruise_speed_mps


def test_standstill_gap_aeb_also_bypasses_the_rate_limit():
    planner = _planner()
    planner.plan(LeadVehicle(distance_m=80.0, range_rate_mps=0.0), 15.0, True, DT)
    decision = planner.plan(LeadVehicle(distance_m=2.0, range_rate_mps=0.0), 15.0, True, DT)
    assert decision.aeb_active is True
    assert "standstill_gap" in decision.reason
    assert decision.target_speed_mps == 0.0


def test_invalid_range_aeb_also_bypasses_the_rate_limit():
    planner = _planner()
    planner.plan(LeadVehicle(distance_m=80.0, range_rate_mps=0.0), 15.0, True, DT)
    decision = planner.plan(
        LeadVehicle(distance_m=float("nan"), range_rate_mps=0.0), 15.0, True, DT
    )
    assert decision.aeb_active is True
    assert decision.reason.startswith("invalid_range")
    assert decision.target_speed_mps == 0.0


def test_a_coasting_aeb_lead_still_stops_immediately():
    """The coast guard must never soften an emergency stop."""
    planner = _planner()
    planner.plan(LeadVehicle(distance_m=80.0, range_rate_mps=0.0), 15.0, True, DT)
    decision = planner.plan(
        LeadVehicle(distance_m=15.0, range_rate_mps=-15.0, frames_since_measurement=3),
        15.0,
        True,
        DT,
    )
    assert decision.aeb_active is True
    assert decision.target_speed_mps == 0.0
    assert "_coast3" in decision.reason


def test_dithering_range_across_the_aeb_boundary_does_not_chatter():
    """One pixel of box-height noise used to swing the command 6.0 <-> 14.9 m/s."""
    rng = random.Random(SEED + 1)
    planner = _planner()
    lim = planner.limits
    previous = None
    for _ in range(1000):
        distance = 26.5 + rng.uniform(-0.6, 0.6)  # straddles the a_req = 5 boundary
        decision = planner.plan(
            LeadVehicle(distance_m=distance, range_rate_mps=-15.0), 15.0, True, DT
        )
        if previous is not None:
            assert abs(decision.target_speed_mps - previous) <= (
                max(lim.max_accel_mps2, lim.emergency_decel_mps2) * DT + 1e-9
            )
        previous = decision.target_speed_mps


def test_comfort_decel_bounds_the_target_when_a_lead_disappears_and_reappears():
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


def test_dropout_ramp_uses_comfort_then_mrm_rate():
    planner = _planner()
    lim = planner.limits
    planner.plan(LeadVehicle(distance_m=60.0, range_rate_mps=0.0), 15.0, True, DT)
    start = planner.previous_target_mps
    first = planner.plan(None, 15.0, perception_valid=False, dt_s=DT)
    assert start - first.target_speed_mps == pytest.approx(lim.max_decel_mps2 * DT, rel=1e-6)
    second = planner.plan(None, 15.0, perception_valid=False, dt_s=DT)
    assert first.target_speed_mps - second.target_speed_mps == pytest.approx(
        lim.max_decel_mps2 * DT, rel=1e-6
    )
    third = planner.plan(None, 15.0, perception_valid=False, dt_s=DT)
    assert second.target_speed_mps - third.target_speed_mps == pytest.approx(
        lim.mrm_decel_mps2 * DT, rel=1e-6
    )


def test_dropout_recovery_resumes_from_the_ramped_value_not_from_cruise():
    planner = _planner()
    planner.plan(None, 15.0, perception_valid=True, dt_s=DT)
    for _ in range(5):
        planner.plan(None, 15.0, perception_valid=False, dt_s=DT)
    ramped = planner.previous_target_mps
    decision = planner.plan(None, 15.0, perception_valid=True, dt_s=DT)
    assert decision.target_speed_mps <= ramped + planner.limits.max_accel_mps2 * DT + 1e-9


def test_unknown_ego_speed_degrades_and_never_guesses():
    planner = _planner()
    decision = planner.plan(LeadVehicle(distance_m=40.0, range_rate_mps=0.0), None, True, DT)
    assert decision.degraded is True
    assert decision.reason == "ego_speed_unavailable"
    assert decision.target_speed_mps == 0.0
    for bad in (float("nan"), float("inf"), -1.0):
        planner.reset()
        assert planner.plan(None, bad, True, DT).reason == "ego_speed_unavailable"


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
    """Regression: ego.source='none' is the shipped default and a PERMANENT state.

    One WARNING per frame at 20 Hz is 72,000 lines an hour, which fills the disk
    and buries every real event.  The condition must still be visible on every
    frame -- in ``SpeedDecision`` and in ``ego_speed_available`` -- just not in
    the log.
    """
    planner = _planner()
    with caplog.at_level(logging.DEBUG, logger="adas.planning.longitudinal"):
        for _ in range(1200):  # 60 s at 20 Hz
            decision = planner.plan(None, None, True, DT)
            assert decision.degraded is True
            assert decision.reason == "ego_speed_unavailable"
            assert planner.ego_speed_available is False
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, "expected 1 WARNING for 1200 frames, got %d" % len(warnings)
    assert "no usable ego speed" in warnings[0].getMessage()


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


def _closing_run(planner, gap0, closing_true, reported_rate, ego_speed, frames, dt=DT,
                 track_id=1, coasting=0):
    """Drive `planner` with a range that really collapses at `closing_true`."""
    out = []
    gap = gap0
    for _ in range(frames):
        lead = LeadVehicle(
            distance_m=gap,
            range_rate_mps=reported_rate,
            track_id=track_id,
            frames_since_measurement=coasting,
        )
        out.append(planner.plan(lead, ego_speed, dt_s=dt))
        gap = max(0.0, gap - closing_true * dt)
    return out


def test_contradicted_rate_channel_still_produces_an_aeb():
    """The exact reviewer scenario: stationary lead, velocity channel stuck at 0.

    Before the cross-check the planner said ``follow_gap`` on all 40 frames and
    the controller commanded brake 0.000 on all 40 -- the arbiter was the only
    thing braking. It now declares AEB.

    The LATENCY is pinned deliberately and is not free: the fit needs a full
    window of evidence, so with a channel that lies from the first frame the AEB
    lands one window (0.95 s) in, at 11.5 m of the original 25 m. An ego at
    15 m/s needs 14 m to stop at 8 m/s^2, so this MITIGATES the stuck-channel
    case, it does not make it survivable -- measured impact speed falls from
    13.7 m/s to 6.6 m/s in the closed loop. Shortening the window is what would
    buy the rest, and the measured grid in ``LongitudinalLimits`` says every
    shorter setting fabricates corrections out of range noise instead. If this
    latency has to come down, the evidence has to get better, not looser.
    """
    planner = LongitudinalPlanner()
    window = planner.limits.range_rate_cross_check_window
    decisions = _closing_run(planner, 25.0, 15.0, 0.0, 15.0, 40)
    assert any(d.range_rate_corrected for d in decisions)
    assert any(d.aeb_active for d in decisions), [d.reason for d in decisions]
    fired = next(i for i, d in enumerate(decisions) if d.aeb_active)
    assert fired <= window, "AEB took %d frames, more than one window" % fired
    assert decisions[fired].target_speed_mps == 0.0
    assert decisions[fired].range_rate_mps == pytest.approx(-15.0, abs=0.2)


def test_cross_check_is_silent_when_the_rate_channel_is_honest():
    planner = LongitudinalPlanner()
    decisions = _closing_run(planner, 60.0, 5.0, -5.0, 15.0, 80)
    assert not any(d.range_rate_corrected for d in decisions)


def test_cross_check_is_silent_on_a_steady_gap():
    planner = LongitudinalPlanner()
    decisions = _closing_run(planner, 45.0, 0.0, 0.0, 15.0, 100)
    assert not any(d.range_rate_corrected for d in decisions)
    assert not any(d.aeb_active for d in decisions)


def test_cross_check_never_implies_worse_than_a_stationary_obstacle():
    """The correction is clamped at -v_ego, so it cannot invent an oncoming lead."""
    planner = LongitudinalPlanner()
    # Range collapsing at 30 m/s while the ego only does 4 m/s: physically this
    # is an approaching object, but the correction path must not act on more than
    # the stationary-obstacle rate.
    decisions = _closing_run(planner, 200.0, 30.0, 0.0, 4.0, 40)
    corrected = [d for d in decisions if d.range_rate_corrected]
    assert corrected, "expected the disagreement to be detected"
    assert all(d.range_rate_mps >= -4.0 - 1e-9 for d in corrected), [
        d.range_rate_mps for d in corrected
    ]


def test_cross_check_ignores_an_isolated_range_jump():
    """One bad range sample must not move a median-of-N estimate."""
    planner = LongitudinalPlanner()
    seen = []
    for i in range(60):
        gap = 40.0 if i != 30 else 4.5  # single-frame collapse and back again
        seen.append(
            planner.plan(
                LeadVehicle(distance_m=gap, range_rate_mps=0.0, track_id=1), 15.0, dt_s=DT
            )
        )
    assert not any(d.range_rate_corrected for d in seen[31:]), (
        "an isolated range jump moved the cross-check"
    )


def test_cross_check_needs_a_full_window():
    planner = LongitudinalPlanner()
    window = planner.limits.range_rate_cross_check_window
    decisions = _closing_run(planner, 60.0, 15.0, 0.0, 15.0, window - 1)
    assert not any(d.range_rate_corrected for d in decisions)
    # ...and fires on the very next frame, once the window is complete.
    assert _closing_run(planner, 60.0 - 0.75 * (window - 1), 15.0, 0.0, 15.0, 1)[
        0
    ].range_rate_corrected


def test_cross_check_restarts_on_a_track_change():
    planner = LongitudinalPlanner()
    gap = 60.0
    corrected = []
    for i in range(40):
        lead = LeadVehicle(distance_m=gap, range_rate_mps=0.0, track_id=i)  # new id each frame
        corrected.append(planner.plan(lead, 15.0, dt_s=DT).range_rate_corrected)
        gap -= 0.75
    assert not any(corrected)


def test_cross_check_restarts_on_a_coasting_track():
    planner = LongitudinalPlanner()
    decisions = _closing_run(planner, 60.0, 15.0, 0.0, 15.0, 40, coasting=1)
    assert not any(d.range_rate_corrected for d in decisions)


def test_cross_check_restarts_after_a_perception_dropout():
    planner = LongitudinalPlanner()
    gap = 60.0
    for _ in range(30):
        planner.plan(LeadVehicle(distance_m=gap, range_rate_mps=0.0, track_id=1), 15.0, dt_s=DT)
        gap -= 0.75
    planner.plan(None, 15.0, perception_valid=False, dt_s=DT)
    d = planner.plan(
        LeadVehicle(distance_m=gap, range_rate_mps=0.0, track_id=1), 15.0, dt_s=DT
    )
    assert not d.range_rate_corrected


def test_cross_check_logs_once_not_once_per_frame(caplog):
    planner = LongitudinalPlanner()
    with caplog.at_level(logging.WARNING, logger="adas.planning.longitudinal"):
        _closing_run(planner, 400.0, 15.0, 0.0, 15.0, 300)
    lines = [r for r in caplog.records if "collapsing" in r.getMessage()]
    assert len(lines) == 1, "%d warnings for one latched condition" % len(lines)


def test_reset_clears_the_cross_check():
    planner = LongitudinalPlanner()
    _closing_run(planner, 400.0, 15.0, 0.0, 15.0, 40)
    planner.reset()
    d = planner.plan(LeadVehicle(distance_m=300.0, range_rate_mps=0.0, track_id=1), 15.0, dt_s=DT)
    assert not d.range_rate_corrected


def test_range_rate_override_preserves_monotonicity():
    """raw_target_speed_mps must be monotone in the OVERRIDE argument too."""
    planner = LongitudinalPlanner()
    for ego in EGO_SPEEDS:
        for distance in (0.0, 3.0, 8.0, 15.0, 30.0, 60.0, 120.0):
            previous = None
            rate = -30.0
            while rate <= 30.0:
                lead = LeadVehicle(distance_m=distance, range_rate_mps=0.0)
                target = planner.raw_target_speed_mps(lead, ego, range_rate_mps=rate)[0]
                if previous is not None:
                    assert target >= previous - 1e-9, (ego, distance, rate)
                previous = target
                rate += 0.25


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
def test_cross_check_tuning_is_validated(kwargs):
    with pytest.raises(ValidationError):
        LongitudinalLimits(**kwargs)


def test_cross_check_is_silent_under_range_noise_with_an_honest_channel(caplog):
    """The gate that matters: a NOISY range must not manufacture a closing rate.

    The first version of this cross-check took the median of the per-frame range
    differences. Differencing multiplies range noise by 1/dt (20x at 20 Hz) and a
    median only divides it by 3, so with an honest rate channel and 0.25 m of
    gaussian range noise it corrected 180 frames in 3000, and at 5 m of noise it
    manufactured AEB frames from a lead that was not closing at all. The fit's own
    standard error is what closes that hole, so this test sweeps the noise.
    """
    for sigma in (0.25, 0.5, 1.0, 2.0, 5.0, 8.0):
        rng = random.Random(SEED + int(sigma * 100))
        planner = LongitudinalPlanner()
        corrected = 0
        aeb = 0
        for _ in range(600):
            noisy = max(0.1, 40.0 + rng.gauss(0.0, sigma))
            decision = planner.plan(
                LeadVehicle(distance_m=noisy, range_rate_mps=0.0, track_id=1), 15.0, dt_s=DT
            )
            corrected += int(decision.range_rate_corrected)
            aeb += int(decision.aeb_active)
        assert corrected == 0, "sigma=%.2f m: %d fabricated corrections" % (sigma, corrected)
        assert aeb == 0, "sigma=%.2f m: %d phantom AEB frames" % (sigma, aeb)


def test_cross_check_still_fires_through_range_noise_when_the_channel_lies():
    """Significance must not be bought by making the check deaf."""
    rng = random.Random(SEED + 7)
    planner = LongitudinalPlanner()
    gap = 60.0
    corrected = 0
    for _ in range(60):
        gap = max(0.5, gap - 15.0 * DT)
        noisy = max(0.1, gap + rng.gauss(0.0, 0.5))
        corrected += int(
            planner.plan(
                LeadVehicle(distance_m=noisy, range_rate_mps=0.0, track_id=1), 15.0, dt_s=DT
            ).range_rate_corrected
        )
    assert corrected > 30, "only %d corrected frames on a 15 m/s lie" % corrected
