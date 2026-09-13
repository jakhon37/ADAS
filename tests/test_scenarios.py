"""Run the safety acceptance library under pytest.

Two kinds of test live here and they are not the same kind of thing:

* ``test_harness_*`` check the HARNESS -- that the plant integrates correctly,
  that the oracle's kinematics agree with hand arithmetic, and that a run is
  deterministic.  These must always pass; if one fails the specification itself
  is broken and nothing below it means anything.
* ``test_scenario`` checks the SYSTEM against the specification.  A failure
  here is a statement about ``adas.control``, not about this file.  Do not
  relax an expectation to make one of these pass: the expectation is the
  specification, and the physics argument for it is printed in the failure
  message.  Change it only by changing ``docs/SAFETY_SPEC.md`` first.

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

from tests.scenarios import oracle as truth  # noqa: E402
from tests.scenarios import plant as pl  # noqa: E402
from tests.scenarios import report as rep  # noqa: E402
from tests.scenarios import scenario as scen  # noqa: E402
from tests.scenarios.library import SCENARIOS  # noqa: E402


# --------------------------------------------------------------------------- #
# Harness self-checks
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
    names = [s.name for s in SCENARIOS]
    assert len(names) == len(set(names))
    for s in SCENARIOS:
        assert s.frames > 0
        assert len(s.physics) > 200, s.name
        assert len(s.summary) > 10, s.name


# --------------------------------------------------------------------------- #
# The specification itself
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def results():
    """Run the whole library once and share it across the tests."""
    return {r.scenario.name: r for r in rep.run_all()}


@pytest.mark.parametrize("name", [s.name for s in SCENARIOS])
def test_scenario(results, name):
    """The system must satisfy the specification for this scenario."""
    result = results[name]
    if result.passed:
        return
    lines = [
        "",
        "SCENARIO %s FAILED" % name,
        "  %s" % result.scenario.summary,
        "",
        "diagnosis:",
    ]
    for f in result.findings:
        lines.append("  %s: %s" % (f.code.upper(), f.detail))
    lines += [
        "",
        "why this outcome is required:",
        "  %s" % result.scenario.physics,
        "",
        "metrics: %s" % json.dumps(result.metrics, sort_keys=True),
    ]
    pytest.fail("\n".join(lines), pytrace=False)


def test_report_json_is_serialisable(results):
    """The machine-readable artifact must round-trip through json."""
    payload = rep.to_json(list(results.values()))
    text = json.dumps(payload, sort_keys=True)
    back = json.loads(text)
    assert back["summary"]["total"] == len(SCENARIOS)
    assert set(back["margins"]) >= {"contact_gap_m", "required_clearance_m"}


def test_report_table_renders(results):
    """The human-readable table must render for a mixed pass/fail run."""
    text = rep.render_table(list(results.values()))
    assert "ADAS LONGITUDINAL SAFETY ACCEPTANCE" in text
    for name in results:
        assert name in text
