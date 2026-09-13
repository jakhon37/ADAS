"""Run the safety acceptance library under pytest.

Three kinds of test live here and they are not the same kind of thing:

* ``test_harness_*`` check the HARNESS -- that the plant integrates correctly,
  that the oracle's kinematics agree with hand arithmetic, that a run is
  deterministic, and that every diagnosis this specification can print is
  actually REACHABLE.  These must always pass; if one fails the specification
  itself is broken and nothing below it means anything.
* ``test_scenario`` checks the SYSTEM against the specification.  A failure
  here is a statement about ``adas.control``, not about this file.  Do not
  relax an expectation to make one of these pass: the expectation is the
  specification, and the physics argument for it is printed in the failure
  message.  Change it only by changing ``docs/SAFETY_SPEC.md`` first.
* ``test_no_new_scenario_failures`` and ``test_baseline_is_current`` check the
  BASELINE -- see below.

The baseline
------------
The arbiter is broken today, so most of ``test_scenario`` is red, and a wall of
red hides a regression as effectively as a wall of green.
``tests/scenarios/baseline.json`` records which scenarios are known to fail and
with which diagnoses.  It is NOT a suppression list and these are NOT xfails:

* every known failure still fails, still prints its full diagnosis, and still
  makes the suite red -- a safety failure that stops being visible has stopped
  being a safety failure;
* its message is prefixed ``KNOWN FAILURE`` so it can be told apart at a glance
  from one prefixed ``NEW FAILURE``;
* ``test_no_new_scenario_failures`` gives the distinction one clean signal, so
  "did I break something?" is answerable without reading 33 tracebacks;
* ``test_baseline_is_current`` fails when a listed scenario starts passing, so
  the file can only ever shrink.

The redesign's target is written down there: the job is done when
``known_failures`` is empty.

Everything runs on CPU with no engine, no camera and no recording, so this file
works in CI and off the target board.

Deselect the whole acceptance suite with ``--deselect tests/test_scenarios.py``
or ``-k "not scenario"`` if you need a green run for unrelated work.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, _ROOT)

from adas.core.models import ControlCommand, SafetyState  # noqa: E402

from tests.scenarios import oracle as truth  # noqa: E402
from tests.scenarios import library as lib  # noqa: E402
from tests.scenarios import plant as pl  # noqa: E402
from tests.scenarios import report as rep  # noqa: E402
from tests.scenarios import scenario as scen  # noqa: E402

ALL_SCENARIOS = rep.all_scenarios()
"""The library plus the independence cases.  The corpus this file grades."""

BASELINE = rep.load_baseline()
"""The committed known-failure list; ``{}`` when the file has not been written."""


# --------------------------------------------------------------------------- #
# Harness self-checks: the plant and the oracle
# --------------------------------------------------------------------------- #


def test_harness_plant_stopping_distance_matches_kinematics():
    """Full braking from 20 m/s must stop in v^2/2a plus the actuator lag.

    v^2 / (2 x 8) = 25.0 m.  The 0.15 s brake rise adds roughly one time
    constant of travel at 20 m/s, about 1.3 m, so the plant should land between
    25 and 28 m.  Outside that band the plant is not the vehicle its constants
    claim, and every avoidability judgement built on it is wrong.
    """
    p = pl.Plant(ego_speed_mps=20.0)
    while p.state.ego_v_mps > 1e-6 and p.state.t_s < 10.0:
        p.step(0.0, 1.0, 0.0)
    assert 25.0 <= p.state.ego_x_m <= 28.0, p.state.ego_x_m


def test_harness_plant_holds_speed_with_no_command():
    """No throttle, no brake, no drag: the speed must not drift."""
    p = pl.Plant(ego_speed_mps=20.0)
    for _ in range(200):
        p.step(0.0, 0.0, 0.0)
    assert p.state.ego_v_mps == pytest.approx(20.0)


def test_harness_plant_is_deterministic():
    """Two identical runs of a noisy scenario must agree bit for bit."""
    noisy = scen.Scenario(
        name="determinism",
        summary="noisy perception",
        physics="determinism check",
        frames=60,
        ego_speed_mps=20.0,
        lead=pl.LeadSpec(40.0, 20.0, pl.lead_constant_speed()),
        perception=pl.PerceptionSpec(range_noise_m=0.8),
        expect=scen.Expectation(),
    )
    a = scen.run(noisy)
    b = scen.run(noisy)
    assert [r.command.brake for r in a.records] == [r.command.brake for r in b.records]
    assert [s.ego_x_m for s in a.history] == [s.ego_x_m for s in b.history]


# --------------------------------------------------------------------------- #
# Harness self-checks: coverage, satisfiability and the fidelity features
# --------------------------------------------------------------------------- #


def test_harness_every_scenario_is_physically_satisfiable():
    """No scenario may demand a clearance no correct system can hold.

    The converse of the rule this library is written under.  Relaxing an
    expectation so the current code passes destroys the specification -- but an
    expectation that is PHYSICALLY UNSATISFIABLE destroys it just as thoroughly,
    because the only way to pass it is to brake before the measurement exists,
    which is the phantom the same specification punishes everywhere else.  A
    scenario like that manufactures a bug report out of the harness's own
    arithmetic.

    Five scenarios were in exactly that state when this test was written --
    ``stationary_15mps_at_15m`` (0.25 m demanded, -0.21 m reachable),
    ``stationary_20mps_at_26m`` (0.10 / -0.30), ``stationary_20mps_at_29m``
    (2.00 / -0.23), ``stationary_25mps_at_42m`` (1.00 / -0.17) and
    ``lane_error_with_closing_lead`` (2.00 / -0.01) -- because they were placed
    against the plant's raw stopping distance rather than against the
    avoidability boundary a compliant system can actually reach.  They have been
    re-placed against ``library.AVOIDABILITY_M``; this test is what stops them
    drifting back.
    """
    bad = []
    for scenario in ALL_SCENARIOS:
        feas = scen.feasibility(scenario)
        if not feas.feasible:
            bad.append("  %-46s %s" % (scenario.name, feas.summary))
    assert not bad, (
        "\n%d scenario(s) demand a clearance that no correct system can reach:\n%s\n\n"
        "Move the obstacle out, or lower the requirement to what is reachable. Do NOT "
        "leave it: a scenario passable only by braking on the stationary prior rewards "
        "the exact defect this harness exists to catch." % (len(bad), "\n".join(bad))
    )


def test_harness_every_scenario_has_decision_slack():
    """A satisfiable scenario must also not be a knife edge.

    ``feasible`` only asks whether the requirement is reachable with ZERO
    decision latency after the first honest measurement.  A case that needs the
    command to go out on that exact frame is technically passable and
    practically a coin toss: it cannot distinguish a system that is wrong from
    one that is one frame slow, and every run of it is a report about the
    scheduler.

    One frame (50 ms) is the floor asserted here.  It is deliberately low --
    boundary cases are supposed to be tight -- but it is not zero.
    """
    tight = []
    for scenario in ALL_SCENARIOS:
        feas = scen.feasibility(scenario)
        if not math.isfinite(feas.affordable_decision_latency_s):
            continue
        if feas.affordable_decision_latency_s < pl.DT_S - 1e-9:
            tight.append(
                "  %-46s %.3f s of slack (%.2f m reachable, %.2f m demanded)"
                % (
                    scenario.name,
                    feas.affordable_decision_latency_s,
                    feas.best_clearance_m,
                    feas.required_clearance_m,
                )
            )
    assert not tight, (
        "\n%d scenario(s) leave less than one frame of decision latency:\n%s\n\n"
        "Move the case out by a metre or two. A boundary case is meant to be tight, "
        "not to require a command on one exact frame." % (len(tight), "\n".join(tight))
    )


def test_harness_fidelity_features_are_exercised():
    """Every fidelity feature the plant offers must be used by some scenario.

    This is the census, and it is committed because the last revision failed it
    silently.  The plant grew sensor noise, range bias, variable frame periods
    and multi-object worlds, and a programmatic count over all 37 scenarios
    found ``range_noise_m > 0`` in 0, ``range_bias_frac != 0`` in 0 and
    ``dt != 0.05`` in 0.  Nothing in the repository said so, because nothing
    counted.

    A capability with no coverage is worse than an absent one: it reads as
    fidelity in the plant's docstrings and delivers none, and the specific
    consequence here was that the arbiter's corroboration logic -- which exists
    because of a false positive measured at +/-0.30 m of range noise -- could
    not be certified by the harness that is supposed to certify it.
    """
    census = lib.coverage_census(ALL_SCENARIOS)
    empty = [feature for feature, names in census.items() if not names]
    assert not empty, (
        "\n%d plant fidelity feature(s) are exercised by NO scenario:\n  %s\n\n%s\n\n"
        "Either add a scenario that uses it, ON a boundary where it matters, or remove "
        "the capability from the plant. A feature with zero coverage is a claim the "
        "harness cannot back."
        % (len(empty), "\n  ".join(empty), lib.render_census(ALL_SCENARIOS))
    )


def test_harness_sensor_noise_actually_perturbs_the_measurements():
    """Seeded noise must be non-zero noise.

    A census counts declarations.  This checks the declaration does something:
    a scenario configured for +/-0.30 m of range noise must actually be handed
    ranges that differ from the truth, with roughly that spread.  Without this
    the census could be satisfied by a field set to a value the sensor ignores,
    which is how a coverage number becomes a lie.
    """
    spec = pl.noisy_perception(range_noise_m=lib.HISTORIC_RANGE_NOISE_M, seed=20240914)
    plant = pl.Plant(
        ego_speed_mps=20.0,
        lead=pl.LeadSpec(40.0, 20.0, pl.lead_constant_speed()),
    )
    sensor = pl.Sensor(spec)
    errors = []
    state = plant.state
    for _ in range(200):
        obs = sensor.observe(state)
        if obs.tracks:
            errors.append(obs.tracks[0].distance_m - state.gap_m)
        state = plant.step(0.0, 0.0, 0.0)
    assert len(errors) > 150, len(errors)
    spread = (sum(e * e for e in errors) / len(errors)) ** 0.5
    assert spread > 0.15, (
        "range noise of %.2f m produced an RMS reported-versus-true error of only "
        "%.4f m; the scenario declares noise the sensor is not applying"
        % (lib.HISTORIC_RANGE_NOISE_M, spread)
    )
    assert spread < 1.0, spread


def test_harness_range_bias_is_systematic_not_noisy():
    """A bias must be the same error every frame, in the same direction.

    The whole reason bias needs its own scenarios is that a corroboration window
    cannot see it: it does not average out.  If the plant applied it as one more
    zero-mean draw the bias scenarios would be duplicate noise scenarios, and
    the census row would be counting nothing.
    """
    bias = 0.10
    plant = pl.Plant(
        ego_speed_mps=20.0,
        lead=pl.LeadSpec(40.0, 20.0, pl.lead_constant_speed()),
    )
    sensor = pl.Sensor(pl.PerceptionSpec(range_bias_frac=bias))
    ratios = []
    state = plant.state
    for _ in range(100):
        obs = sensor.observe(state)
        if obs.tracks:
            ratios.append(obs.tracks[0].distance_m / state.gap_m)
        state = plant.step(0.0, 0.0, 0.0)
    assert ratios
    assert max(ratios) - min(ratios) < 1e-9, (
        "the bias varies frame to frame (%.6f to %.6f); it is noise, not a bias"
        % (min(ratios), max(ratios))
    )
    assert abs(ratios[0] - (1.0 + bias)) < 1e-9, ratios[0]


def test_harness_variable_dt_really_varies_the_step():
    """The dt schedule must reach the plant, not just sit in the config.

    ``frame_overrun_during_stationary_approach`` names three measured
    overruns -- 187.7 ms, 174.3 ms and a 100 ms dropped frame -- and the case
    means nothing unless the world really advances by those durations.
    """
    scenario = lib.by_name("frame_overrun_during_stationary_approach")
    result = scen.run(scenario)
    seen = {round(s.dt_s, 4) for s in result.history}
    for frame, duration in lib.OVERRUN_FRAMES.items():
        assert round(duration, 4) in seen, (
            "frame %d was configured to take %.4f s and no step of that duration "
            "reached the plant; the schedule is being ignored. Durations seen: %s"
            % (frame, duration, sorted(seen))
        )
    elapsed = sum(s.dt_s for s in result.history[1:]) + result.history[0].dt_s
    nominal = len(result.history) * pl.DT_S
    assert elapsed > nominal + lib.OVERRUN_LOST_S - 1e-6, (elapsed, nominal)


@pytest.mark.parametrize(
    "name",
    sorted(
        n
        for feature, names in lib.coverage_census(ALL_SCENARIOS).items()
        for n in names
        if feature.startswith(("range_noise", "lateral_noise", "box_noise", "ego_speed_noise"))
    ),
)
def test_harness_noisy_scenarios_are_deterministic(name):
    """Every noisy scenario must produce identical bytes on two runs.

    Noise in an acceptance harness is only useful if it is reproducible: a
    scenario that fails one run in ten is a scenario people re-run until it
    passes.  The plant draws from a counter-based stream keyed by
    ``(seed, frame, stream, object)``, so this also guards the property that
    makes that worth having -- the draw does not depend on call order, on how
    many objects are in the scene, or on anything that happened earlier.
    """
    scenario = lib.by_name(name)
    a = scen.run(scenario)
    b = scen.run(scenario)
    assert [r.command.brake for r in a.records] == [r.command.brake for r in b.records]
    assert [r.command.throttle for r in a.records] == [r.command.throttle for r in b.records]
    assert [s.ego_x_m for s in a.history] == [s.ego_x_m for s in b.history]
    assert sorted(f.code for f in a.findings) == sorted(f.code for f in b.findings)


def test_harness_out_of_lane_objects_are_genuinely_out_of_the_path():
    """The out-of-lane cases must contain nothing the ego could ever hit.

    This is the satisfiability argument for demanding NO braking in them, and it
    has to be checked rather than asserted in prose: if the lateral offset were
    small enough for the footprints to overlap, "do not brake" would be a demand
    to ignore a real hazard, and the scenario would be rewarding the opposite
    defect.
    """
    names = lib.coverage_census(ALL_SCENARIOS)["out-of-lane object"]
    assert names, "no out-of-lane coverage at all"
    for name in names:
        scenario = lib.by_name(name)
        result = scen.run(scenario)
        for state in result.history:
            for obj in state.objects:
                if not obj.present or obj.gap_m <= 0.0:
                    continue
                if abs(obj.lateral_m) <= pl.LANE_HALF_WIDTH_M + obj.half_width_m:
                    continue  # this one IS in the ego lane on purpose
                assert not obj.in_ego_path, (
                    "%s: object %d is %.2f m off the lane centre at frame %d and the "
                    "plant still calls it in-path; the scenario's 'do not brake' "
                    "expectation would be wrong"
                    % (name, obj.index, obj.lateral_m, state.frame)
                )


def test_harness_multi_object_worlds_reach_the_plant():
    """A scenario's extra objects must appear in the true world state.

    ``Plant`` accepted ``others`` and ``WorldState.objects`` existed for two
    revisions while ``Scenario`` had no field for them, so no scenario could
    ever put a second object in the world: a capability with a structural zero.
    This asserts the plumbing, so that a refactor cannot quietly restore the
    zero and leave the multi-object scenarios silently single-object.
    """
    if not lib.scenario_supports_multi_object():
        pytest.fail(
            "Scenario still has no '%s' field, so the two multi-object cases cannot be "
            "built:\n  %s" % (lib.MULTI_OBJECT_FIELD, lib.MULTI_OBJECT_BLOCKED_REASON)
        )
    names = lib.coverage_census(ALL_SCENARIOS)["multiple objects"]
    assert names, "Scenario supports multiple objects and no scenario uses them"
    for name in names:
        scenario = lib.by_name(name)
        expected = 1 + len(getattr(scenario, lib.MULTI_OBJECT_FIELD))
        if scenario.lead is None:
            expected -= 1
        result = scen.run(scenario)
        assert len(result.history[0].objects) == expected, (
            "%s declares %d object(s) and the plant built %d"
            % (name, expected, len(result.history[0].objects))
        )


def test_harness_oracle_required_decel_matches_hand_arithmetic():
    """A stationary obstacle at 30 m with the ego at 20 m/s.

    Keeping 2.0 m of clearance leaves 28 m of usable closure, so the required
    constant deceleration is 20^2 / (2 x 28) = 7.143 m/s^2.
    """
    need = truth.required_decel_mps2(
        gap0_m=30.0, ego_v_mps=20.0, lead_v_mps=0.0, lead_a_mps2=0.0
    )
    assert need == pytest.approx(400.0 / 56.0, abs=1e-3)


def test_harness_oracle_reports_no_requirement_when_not_closing():
    """A lead keeping pace can never require braking, at any range."""
    for gap in (5.0, 20.0, 52.0, 100.0):
        need = truth.required_decel_mps2(
            gap0_m=gap, ego_v_mps=20.0, lead_v_mps=20.0, lead_a_mps2=0.0
        )
        assert need == 0.0, gap


def test_harness_oracle_reports_unavoidable_when_it_is():
    """20 m/s into a wall 5 m away cannot be survived with 8 m/s^2."""
    need = truth.required_decel_mps2(
        gap0_m=5.0, ego_v_mps=20.0, lead_v_mps=0.0, lead_a_mps2=0.0
    )
    assert math.isinf(need)


def test_harness_oracle_min_gap_is_exact_for_a_braking_lead():
    """Hand-check the closed form against a lead braking at 6 m/s^2.

    Ego 20 m/s at a 15 m gap, lead 20 m/s decelerating at 6 m/s^2, ego braking
    at 8 m/s^2.  Relative speed starts at zero and the ego decelerates harder,
    so the gap only ever grows until the ego stops at t = 2.5 s.  The lead stops
    at t = 3.33 s having covered 33.33 m; the ego covers 25.0 m.  The minimum
    gap is therefore the initial 15 m.
    """
    got = truth.min_gap_under_constant_decel(15.0, 20.0, 20.0, -6.0, 8.0)
    assert got == pytest.approx(15.0, abs=1e-6)


def test_harness_scenario_library_is_well_formed():
    """Every scenario must carry a unique name and a real physics argument."""
    names = [s.name for s in ALL_SCENARIOS]
    assert len(names) == len(set(names))
    for s in ALL_SCENARIOS:
        assert s.frames > 0
        assert len(s.physics) > 200, s.name
        assert len(s.summary) > 10, s.name


# --------------------------------------------------------------------------- #
# Harness self-checks: the new judgement layer
# --------------------------------------------------------------------------- #


def test_harness_jerk_limits_are_derived_not_borrowed():
    """The jerk ceilings must be the ones this spec derives, not the arbiter's.

    The emergency ceiling is ``full authority / a human panic-brake rise time``
    = 8.0 / 0.4 = 20 m/s^3, and the comfort ceiling is the top of the band a
    seated occupant does not register, 2.5 m/s^3.  Neither is read from
    ``adas.control``; this test pins that, because tuning a limit to whatever
    the code already does is the exact failure mode this harness exists to
    prevent.
    """
    assert truth.EMERGENCY_JERK_MPS3 == pytest.approx(
        pl.DEFAULT_PLANT.max_brake_decel_mps2 / 0.4
    )
    assert truth.COMFORT_JERK_MPS3 == 2.5
    assert truth.jerk_limit_mps3(True) == truth.EMERGENCY_JERK_MPS3
    assert truth.jerk_limit_mps3(False) == truth.COMFORT_JERK_MPS3


def test_harness_jerk_series_uses_the_real_step_duration():
    """A demand step across a 200 ms overrun is a quarter of the jerk of a 50 ms one.

    A frame that took 200 ms really did hold the previous command for 200 ms, so
    dividing every difference by the nominal period would report four times the
    jerk the occupant felt.  The plant already measures the true step; the jerk
    series must use it.
    """
    def rec(frame, brake, dt):
        return scen.FrameRecord(
            frame=frame,
            t_s=frame * dt,
            true=pl.WorldState(frame=frame, ego_v_mps=20.0, dt_s=dt),
            perception_ok=True,
            detected=False,
            plan_reason="",
            plan_target_mps=20.0,
            raw_command=ControlCommand(0.0, brake, 0.0),
            command=ControlCommand(0.0, brake, 0.0),
            safety_state=SafetyState.NOMINAL,
        )

    fast = scen.commanded_jerk_series([rec(0, 0.0, 0.05), rec(1, 0.25, 0.05)])
    slow = scen.commanded_jerk_series([rec(0, 0.0, 0.05), rec(1, 0.25, 0.20)])
    assert fast[1] == pytest.approx(0.25 * 8.0 / 0.05)
    assert slow[1] == pytest.approx(0.25 * 8.0 / 0.20)


def test_harness_self_report_policy_matches_the_stated_rule():
    """Own output and own proprioception are assertable; exteroception is not.

    In particular the arbiter's ``_arbiter_induced`` / ``_arbiter_commanded``
    suffixes must NOT exempt a finding: they attribute it, and an occupant does
    not feel an attribution.
    """
    for code in (
        "jerk_32.2_above_15.0",
        "jerk_32.2_above_15.0_arbiter_induced",
        "measured_decel_9.10_above_8.00_unjustified",
        "measured_decel_9.10_above_8.00_arbiter_commanded",
        "command_throttle_brake_conflict",
        "command_not_finite",
        "decision_path_exception",
    ):
        assert scen.is_assertable_self_report(code), code
    for code in (
        "gap_8.10m_below_absolute_min",
        "ttc_1.20s_below_warn_threshold",
        "perception_dropout_7:source",
        "camera_uncalibrated",
        "range_jump_track1",
        "timing_dt_above_max_0.190s",
        "lane_departure_1.10m_above_0.80m",
        "plan_accel_3.20_above_2.50_arbiter_induced",
    ):
        assert not scen.is_assertable_self_report(code), code
    assert scen.self_report_family("jerk_32.2_above_15.0_arbiter_induced") == "jerk_above_limit"


def test_harness_stack_spec_rejects_unknown_modes():
    """A typo in a stack mode must not silently fall back to the real stack."""
    with pytest.raises(ValueError):
        scen.StackSpec(planner="mostly_real")
    with pytest.raises(ValueError):
        scen.StackSpec(arbiter="advisory")
    assert scen.DEFAULT_STACK.is_default
    assert not scen.ARBITER_ONLY_STACK.is_default
    assert scen.ARBITER_ONLY_STACK.arbiter_only


def test_harness_blinding_the_planner_does_not_blind_the_arbiter():
    """PLANNER_BLIND must remove the planner's objects and nothing else.

    If it also emptied the arbiter's track list the independence scenarios would
    be measuring nothing, and they would pass by being vacuous -- which is
    exactly how ``reid_during_steady_follow`` used to pass.
    """
    blind = scen.run(_stationary_case("blind_probe", scen.ARBITER_ONLY_STACK))
    assert max(r.raw_decel_mps2 for r in blind.records) == pytest.approx(0.0), (
        "the blinded primary path braked anyway, so the planner still sees the car"
    )
    assert max(r.arbiter_decel_mps2 for r in blind.records) >= truth.EMERGENCY_DECEL_MPS2, (
        "the arbiter did not brake, so it never received the tracks either"
    )


def test_harness_bypassing_the_arbiter_actuates_the_primary_command():
    """ARBITER_BYPASS must actuate the primary path byte for byte.

    An arbiter whose answer leaked through would make every primary-path claim
    in this suite unfalsifiable.
    """
    result = scen.run(_stationary_case("bypass_probe", scen.PRIMARY_ONLY_STACK))
    for r in result.records:
        assert r.command.brake == r.raw_command.brake, r.frame
        assert r.command.throttle == r.raw_command.throttle, r.frame
    assert any(
        r.arbiter_command is not None and r.arbiter_command.brake != r.raw_command.brake
        for r in result.records
    ), "the arbiter never disagreed, so this run proves nothing about bypassing it"


# --------------------------------------------------------------------------- #
# Harness self-checks: every diagnosis must be reachable
# --------------------------------------------------------------------------- #
#
# The first version of this harness never emitted collided, collided_unavoidable,
# clearance, missed_intervention, late_intervention, no_response or
# lane_departure against EITHER known-broken commit.  Half the specification was
# decorative and nothing said so.  A diagnosis that cannot fire is worse than a
# missing one, because it reads as coverage.


def _stationary_case(name: str, stack: scen.StackSpec, gap_m: float = 40.0) -> scen.Scenario:
    """A stopped car ``gap_m`` ahead of an ego at 20 m/s, run on ``stack``."""
    return scen.Scenario(
        name=name,
        summary="stopped vehicle ahead, used as a fixed physical demand",
        physics="20 m/s into a stopped car at %.0f m. Keeping 2.0 m of clearance leaves "
        "%.0f m of usable closure so %.2f m/s^2 is required, against 8.0 m/s^2 of "
        "authority: an unambiguous emergency with margin to spare. Used as a constant "
        "demand while the stack under it is varied." % (
            gap_m, gap_m - 2.0, 400.0 / (2.0 * (gap_m - 2.0))
        ),
        frames=140,
        ego_speed_mps=20.0,
        lead=pl.LeadSpec(gap_m, 0.0, pl.lead_stationary(), label="stopped car"),
        stack=stack,
        expect=scen.Expectation(must_intervene=True),
    )


NULL_STACK = scen.StackSpec(planner=scen.PLANNER_BLIND, arbiter=scen.ARBITER_BYPASS)
"""Nothing in the stack can see the obstacle and nothing can override that.

Not an acceptance case -- no system is required to survive having both halves
removed.  It is the harness's own control: a run in which the vehicle
demonstrably drives into a stopped car, so that a specification which stays
silent about it can be shown to be broken.
"""


def test_harness_collision_diagnoses_fire_when_nothing_brakes():
    """With the whole decision path removed, the collision half must fire.

    Both layers are gone, the car ahead is stopped, and the physics is
    arithmetic: 40 m at 20 m/s with no braking is contact at t = 2.0 s.  If this
    run produces no collision diagnosis then the detector is broken, and every
    green ``must_intervene`` scenario in the library is meaningless.
    """
    result = scen.run(_stationary_case("null_stack_probe", NULL_STACK))
    codes = {f.code for f in result.findings}
    assert result.verdict.collided, result.metrics
    assert max(r.commanded_decel_mps2 for r in result.records) < truth.NEGLIGIBLE_DECEL_MPS2
    assert "collided" in codes, codes
    assert "missed_intervention" in codes, codes


def test_harness_no_response_fires_for_an_ignored_sub_emergency_approach():
    """A closing lead that never becomes an emergency, and a system that sleeps.

    Ego 20 m/s, lead 30 m ahead holding 14 m/s, for 3 s.  Closing at 6 m/s the
    gap runs 30 m -> 12 m and the requirement to keep 2.0 m of clearance peaks
    at 6^2 / (2 x 10) = 1.8 m/s^2 -- a real requirement, above the 1.0 m/s^2
    hazard floor, and below the 3.0 m/s^2 comfort line, so no emergency ever
    exists and the ``missed_intervention`` branch above cannot fire.  Doing
    NOTHING here is still wrong, and ``no_response`` is the only diagnosis that
    says so.
    """
    scenario = scen.Scenario(
        name="no_response_probe",
        summary="a slower lead closed on and ignored entirely",
        physics="Ego 20 m/s, lead 30 m ahead at a steady 14 m/s. The closing rate is "
        "6 m/s throughout, so over 3 s the gap runs from 30 m to 12 m and the constant "
        "deceleration needed to keep the 2.0 m standstill clearance rises from "
        "36/(2 x 28) = 0.64 m/s^2 to 36/(2 x 10) = 1.8 m/s^2. That is above the "
        "1.0 m/s^2 floor below which nothing is a hazard and below the 3.0 m/s^2 "
        "comfort line, so this is an ordinary following correction and never an "
        "emergency. A system that commands literally nothing has still failed.",
        frames=60,
        ego_speed_mps=20.0,
        lead=pl.LeadSpec(30.0, 14.0, pl.lead_constant_speed(), label="slower car"),
        stack=NULL_STACK,
        expect=scen.Expectation(must_intervene=True),
    )
    result = scen.run(scenario)
    codes = {f.code for f in result.findings}
    assert not result.verdict.emergency, result.metrics
    assert "no_response" in codes, codes


def test_harness_missed_intervention_fires_for_a_token_brake():
    """A system that brakes, but never at emergency grade, must be caught.

    ``no_response`` covers doing nothing at all.  ``missed_intervention`` covers
    the more dangerous case -- a real brake demand that is simply too small --
    and it needs its own proof of life because the two are different branches.
    """
    scenario = _stationary_case("token_brake_probe", NULL_STACK)
    result = scen.run(scenario)
    # Replace the (empty) demand with a token 2 m/s^2 that never reaches
    # emergency grade, and re-judge the SAME run.
    for r in result.records:
        r.command = ControlCommand(0.0, 0.25, 0.0)
    findings = scen.evaluate(scenario, result.records, result.verdict, [])
    codes = {f.code for f in findings}
    assert "missed_intervention" in codes, codes
    assert "no_response" not in codes, codes


def test_harness_unavoidable_contact_is_reported_as_mis_specified():
    """A scenario that cannot be survived must say so, not blame the system.

    20 m/s at a stopped car 4 m away needs 50 m/s^2.  The right diagnosis is
    ``collided_unavoidable`` -- a statement about the SCENARIO -- and a harness
    that reported plain ``collided`` here would be manufacturing a bug report
    out of its own arithmetic.
    """
    result = scen.run(_stationary_case("unavoidable_probe", NULL_STACK, gap_m=4.0))
    codes = {f.code for f in result.findings}
    assert result.verdict.avoidable is False, result.metrics
    assert "collided_unavoidable" in codes, codes


def test_harness_clearance_and_lateral_diagnoses_fire():
    """``clearance`` and ``lane_departure`` must both be reachable.

    The demand is 8.0 m from 40 m, and it is deliberately SATISFIABLE: the
    avoidability boundary at 20 m/s is 31.30 m, so 8.70 m is reachable and a
    correct system would hold it.  The probe removes the decision path
    (``NULL_STACK``) so that nothing brakes and the run drives into the car,
    which is what makes ``clearance`` fire.

    It used to ask for 10.0 m, which is 1.30 m more than any system can hold
    from 40 m, and the harness's own feasibility check now -- correctly --
    answers an impossible demand with ``infeasible_clearance`` and declines to
    blame the system.  That is the right behaviour and this probe was the wrong
    way to exercise it; ``test_harness_every_scenario_is_physically_satisfiable``
    covers the impossible case from the other side.
    """
    scenario = scen.Scenario(
        name="clearance_probe",
        summary="a stopped car driven into with the decision path removed",
        physics="Ground truth is checked directly: with the planner blind and the "
        "arbiter bypassed nothing brakes, so the ego reaches the stopped car while "
        "sitting 1.2 m off the lane centre. A specification that asks for 8.0 m of "
        "clearance -- of the 8.70 m that is physically reachable from 40 m at 20 m/s "
        "-- and 0.5 m of lane discipline must report both, from the plant's own state "
        "and not from anything the system said about itself.",
        frames=140,
        ego_speed_mps=20.0,
        lead=pl.LeadSpec(40.0, 0.0, pl.lead_stationary(), label="stopped car"),
        initial_lateral_offset_m=1.2,
        stack=NULL_STACK,
        expect=scen.Expectation(
            min_clearance_m=8.0, max_abs_lateral_offset_m=0.5, no_collision=False
        ),
    )
    codes = {f.code for f in scen.run(scenario).findings}
    assert "infeasible_clearance" not in codes, codes
    assert "clearance" in codes, codes
    assert "lane_departure" in codes, codes


def test_harness_pedal_conflict_and_jerk_diagnoses_fire():
    """Both pedals at once, and a one-frame full-authority step, must be caught.

    Judged on hand-built frames rather than on a run, because no mode of this
    stack emits both pedals -- which is the correct answer for the stack and
    the wrong reason to leave the detector untested.
    """
    scenario = _stationary_case("pedal_probe", scen.DEFAULT_STACK)
    base = scen.run(scenario)
    records = base.records[:20]
    records[5].command = ControlCommand(0.6, 0.6, 0.0)
    records[10].command = ControlCommand(0.0, 1.0, 0.0)
    records[11].command = ControlCommand(0.0, 0.0, 0.0)
    codes = {f.code for f in scen.evaluate(scenario, records, base.verdict, [])}
    assert "pedal_conflict" in codes, codes
    assert "excess_jerk" in codes, codes


def test_harness_sub_emergency_band_is_policed():
    """A steady 3.0 m/s^2 with nothing in the world must be a finding.

    3.0 m/s^2 is below the 3.5 m/s^2 emergency threshold, so the phantom test
    cannot see it, and it is twice the 1.5 m/s^2 a headway law may use.  This is
    the band a phantom brake retreats into when the AEB path is 'fixed'.
    """
    scenario = scen.Scenario(
        name="sub_emergency_probe",
        summary="empty road, steady sub-emergency brake",
        physics="Nothing is in the world at all, so the true collision-avoidance "
        "requirement is identically zero on every frame and the justified ceiling is "
        "the 1.5 m/s^2 headway allowance. A steady 3.0 m/s^2 is twice that and below "
        "the 3.5 m/s^2 emergency threshold, so it is invisible to any AEB-grade test "
        "while still dragging the vehicle down the road.",
        frames=60,
        ego_speed_mps=25.0,
        lead=None,
        expect=scen.Expectation(),
    )
    base = scen.run(scenario)
    for r in base.records:
        r.command = ControlCommand(0.0, 3.0 / 8.0, 0.0)
    findings = scen.evaluate(scenario, base.records, base.verdict, [])
    codes = {f.code for f in findings}
    assert "unwarranted_brake" in codes, codes
    assert "phantom_intervention" not in codes, (
        "3.0 m/s^2 is below emergency grade; the sub-emergency finding must carry it"
    )


def test_harness_every_finding_code_is_reachable_or_named():
    """No diagnosis may exist without something in this file able to produce it.

    The check is deliberately crude -- every code the corpus and the probes
    above can emit, unioned, must cover every code named in the report's own
    vocabulary.  It exists so that adding a ``Finding`` with no way to trigger
    it is a test failure at the moment it is added, rather than a discovery two
    rounds of fixes later.
    """
    emitted = set()
    for r in rep.run_all():
        emitted |= {f.code for f in r.findings}
    reachable_by_probe = {
        "collided",
        "collided_unavoidable",
        "no_response",
        "missed_intervention",
        "clearance",
        "lane_departure",
        "pedal_conflict",
        "excess_jerk",
        "unwarranted_brake",
    }
    unreached = COLLISION_HALF - (emitted | reachable_by_probe)
    assert not unreached, (
        "these collision diagnoses can be printed but nothing here can make them "
        "fire, so they are decoration: %s" % sorted(unreached)
    )


COLLISION_HALF = {
    "collided",
    "collided_unavoidable",
    "clearance",
    "missed_intervention",
    "late_intervention",
    "no_response",
    "lane_departure",
}
"""The half of the specification that never fired in the first version."""


# --------------------------------------------------------------------------- #
# The specification itself
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def results():
    """Run the whole corpus once and share it across the tests."""
    return {r.scenario.name: r for r in rep.run_all()}


def _failure_message(result, header: str, extra: str = "") -> str:
    lines = ["", "%s: %s" % (header, result.scenario.name), "  %s" % result.scenario.summary]
    if not result.scenario.stack.is_default:
        lines.append("  stack: %s" % result.scenario.stack.label)
    lines += ["", "diagnosis:"]
    for f in result.findings:
        lines.append("  %s: %s" % (f.code.upper(), f.detail))
    lines += ["", "why this outcome is required:", "  %s" % result.scenario.physics]
    if extra:
        lines += ["", extra]
    lines += ["", "metrics: %s" % json.dumps(result.metrics, sort_keys=True)]
    return "\n".join(lines)


@pytest.mark.parametrize("name", [s.name for s in ALL_SCENARIOS])
def test_scenario(results, name):
    """The system must satisfy the specification for this scenario.

    A scenario in ``baseline.json`` still fails here -- loudly, with its full
    diagnosis.  The only difference is the header, which says whether this is
    breakage the redesign already knows about or breakage that appeared today.
    """
    result = results[name]
    codes = sorted({f.code for f in result.findings})
    if result.passed:
        return
    known = BASELINE.get(name)
    if known is None:
        pytest.fail(
            _failure_message(
                result,
                "NEW FAILURE -- SCENARIO",
                "This scenario is not in tests/scenarios/baseline.json, so this is a "
                "REGRESSION introduced since the baseline was recorded. Fix the system. "
                "Adding the scenario to the baseline is not a fix.",
            ),
            pytrace=False,
        )
    worse = sorted(set(codes) - set(known))
    if worse:
        pytest.fail(
            _failure_message(
                result,
                "WORSENED -- SCENARIO",
                "This scenario is in the baseline, but with %s. It has now also failed "
                "with %s, which is new breakage inside a known-broken case."
                % (", ".join(known), ", ".join(worse)),
            ),
            pytrace=False,
        )
    pytest.fail(
        _failure_message(
            result,
            "KNOWN FAILURE -- SCENARIO",
            "Recorded in tests/scenarios/baseline.json with exactly these diagnoses. "
            "This is one of the %d defects the arbiter redesign exists to remove; the "
            "redesign is finished when that file is empty. It is reported here rather "
            "than skipped because a safety failure that stops being visible has "
            "stopped being a safety failure." % len(BASELINE),
        ),
        pytrace=False,
    )


def test_no_new_scenario_failures(results):
    """One clean signal for "did anything get worse today?".

    Without this, the answer is buried in a wall of expected red.  This test
    passes while the known defects are exactly the known defects, and fails the
    moment a scenario fails that the baseline did not predict.
    """
    if not BASELINE:
        pytest.skip("no baseline recorded yet: run report.py --write-baseline")
    split = rep.classify_against_baseline(list(results.values()))
    problems = []
    for name, codes in sorted(split["new"].items()):
        problems.append("  NEW       %-40s %s" % (name, ", ".join(codes)))
    for name, codes in sorted(split["worsened"].items()):
        problems.append("  WORSENED  %-40s also fails with %s" % (name, ", ".join(codes)))
    if problems:
        pytest.fail(
            "\n".join(
                [
                    "",
                    "%d scenario(s) fail in a way the committed baseline did not predict."
                    % len(problems),
                    "The known defects are listed in tests/scenarios/baseline.json; these",
                    "are not among them, so something regressed.",
                    "",
                ]
                + problems
                + [
                    "",
                    "Do not add these to the baseline to make this pass. The baseline is "
                    "the redesign's target and it is only allowed to shrink.",
                ]
            ),
            pytrace=False,
        )


def test_baseline_is_current(results):
    """A scenario that has started passing must be removed from the baseline.

    This is the ratchet.  Without it the baseline decays into a list of things
    that used to be broken, and it stops being usable as the redesign's
    definition of done.
    """
    if not BASELINE:
        pytest.skip("no baseline recorded yet: run report.py --write-baseline")
    split = rep.classify_against_baseline(list(results.values()))
    if split["fixed"]:
        pytest.fail(
            "\n".join(
                [
                    "",
                    "%d scenario(s) now PASS but are still listed in "
                    "tests/scenarios/baseline.json:" % len(split["fixed"]),
                ]
                + ["  %s" % n for n in sorted(split["fixed"])]
                + [
                    "",
                    "Good news, and the file has to say so. Re-record it with:",
                    "  python -m tests.scenarios.report --write-baseline",
                ]
            ),
            pytrace=False,
        )
    missing = sorted(set(BASELINE) - {s.name for s in ALL_SCENARIOS})
    assert not missing, (
        "baseline.json names scenarios that no longer exist: %s. A scenario cannot be "
        "retired by deleting it while it is still failing." % missing
    )


def test_report_json_is_serialisable(results):
    """The machine-readable artifact must round-trip through json."""
    payload = rep.to_json(list(results.values()))
    text = json.dumps(payload, sort_keys=True)
    back = json.loads(text)
    assert back["summary"]["total"] == len(ALL_SCENARIOS)
    assert set(back["margins"]) >= {
        "contact_gap_m",
        "required_clearance_m",
        "headway_decel_allowance_mps2",
        "comfort_jerk_mps3",
        "emergency_jerk_mps3",
    }
    assert set(back["baseline"]) == {"new", "known", "worsened", "fixed"}


def test_report_table_renders(results):
    """The human-readable table must render for a mixed pass/fail run."""
    text = rep.render_table(list(results.values()))
    assert "ADAS LONGITUDINAL SAFETY ACCEPTANCE" in text
    for name in results:
        assert name in text
