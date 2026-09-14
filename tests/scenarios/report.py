"""Reporting: a readable pass/fail table and a machine-readable JSON artifact.

Run the whole library and print the table::

    python -m tests.scenarios.report

Write the JSON artifact as well::

    python -m tests.scenarios.report --json build/safety_acceptance.json

Show the physics justification for every scenario, passing or failing::

    python -m tests.scenarios.report --verbose

Rewrite the committed baseline of known-failing scenarios (see
:data:`BASELINE_PATH` and ``docs/SAFETY_SPEC.md``)::

    python -m tests.scenarios.report --write-baseline

The table is the human deliverable and the JSON is the machine one; both carry
the same diagnoses, so a CI job can diff two runs and a person can read one.

The BASELINE is a third thing and it is not a suppression list.  Every scenario
in it still fails, still prints its diagnosis, and still turns the suite red.
What it buys is the distinction between "this is the known breakage the redesign
exists to fix" and "something got worse today", which a bare count of 29 red
tests cannot express.  It is also the redesign's acceptance criterion, written
down: the job is finished when the baseline is empty.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

from tests.scenarios import oracle as truth
from tests.scenarios import scenario as scen
from tests.scenarios.library import SCENARIOS

SPEC_VERSION = "1.2"
"""Bumped whenever a margin convention or an authority limit changes.

A JSON artifact from a different spec version is not comparable with this one.

1.1 added the four assertions that 1.0 was missing -- bounded jerk, the
throttle/brake conflict, the system's own assertable self-reports, and
proportionality across the sub-emergency band -- and the degradable stack.

1.2 fixed the JUDGEMENT LAYER, which is why it is not comparable with 1.1:

* Avoidability is judged against the vehicle AND the sensor the system actually
  drives -- sense latency, rate observability, actuation dead time, brake rise
  -- and against a jerk-limited brake demand rather than an instantaneous step,
  because a step is 160 m/s^3 and this same specification forbids it.  The
  avoidability boundary for a stationary obstacle therefore moved from
  6.67 / 14.70 / 25.85 / 40.13 m to 9.37 / 18.77 / 31.30 / 46.95 m at
  10 / 15 / 20 / 25 m/s, and five scenarios turned out to be placed inside it.
* The plant STOPS AT CONTACT and records the impact speed.
* ``phantom_intervention`` covers an actuated unwarranted brake only;
  entering a state while commanding nothing is ``unwarranted_authority_state``.
* ``excess_jerk`` assesses only a RISING demand, and takes its band from the
  demand rather than from whether the oracle's causal requirement had already
  become visible.
* ``forbid_emergency_intervention`` is read.  It was set on eight scenarios and
  asserted nothing.
* Every scenario is now checked for SATISFIABILITY against a reference
  controller; see ``--reference`` and ``--budgets``.
"""

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.json")
"""The committed list of scenarios that are known to fail today.

One entry per failing scenario, holding the sorted finding codes it fails with.
See the module docstring for what it is and is not.
"""


def load_baseline(path: str = BASELINE_PATH) -> Dict[str, List[str]]:
    """Read the committed baseline, or return ``{}`` when there is not one yet.

    Returns:
        Mapping of scenario name to the sorted finding codes recorded for it.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        payload = json.load(handle)
    return {k: sorted(v) for k, v in payload.get("known_failures", {}).items()}


def build_baseline(results: Sequence[scen.ScenarioResult]) -> Dict[str, object]:
    """The baseline document for this run, provenance block included.

    The ``generated`` block comes from :func:`tests.scenarios.baseline_header`,
    which fingerprints the system under test and the corpus.  It used to be
    written by hand and dropped silently on every regeneration, which is the
    defect the committed file's own ``staleness_check`` text described and asked
    to have fixed here.  A baseline with no provenance is worse than none: it
    compares a run against an unknown pair of (arbiter, corpus) and reports the
    difference as if it meant something.
    """
    known = {
        r.scenario.name: sorted({f.code for f in r.findings})
        for r in results
        if not r.passed
    }
    codes: Dict[str, int] = {}
    for entry in known.values():
        for code in entry:
            codes[code] = codes.get(code, 0) + 1
    from tests.scenarios import baseline_header

    return {
        "spec_version": SPEC_VERSION,
        "generated": baseline_header([r.scenario.name for r in results], known),
        "purpose": (
            "Scenarios that FAIL against the arbiter as committed today. This file "
            "is the redesign's target: the work is done when 'known_failures' is "
            "empty. It is NOT a suppression list -- every scenario named here still "
            "fails, still prints its diagnosis and still makes the suite red. Its "
            "only job is to let pytest say which failures are NEW."
        ),
        "how_to_regenerate": "python -m tests.scenarios.report --write-baseline",
        "rules": [
            "Adding a name here is an admission of a live safety defect. Say why in "
            "the commit message.",
            "Removing a name is the only allowed way to make this file smaller, and "
            "it must be because the scenario now PASSES.",
            "Never edit a scenario's expectation to get it out of this file. The "
            "expectation is the specification.",
        ],
        "totals": {"scenarios": len(results), "known_failures": len(known)},
        "diagnosis_counts": dict(sorted(codes.items())),
        "known_failures": known,
    }


def write_baseline(results: Sequence[scen.ScenarioResult], path: str = BASELINE_PATH) -> None:
    """Write the baseline document, sorted and indented so diffs are readable."""
    with open(path, "w") as handle:
        json.dump(build_baseline(results), handle, indent=2, sort_keys=True)
        handle.write("\n")


def classify_against_baseline(
    results: Sequence[scen.ScenarioResult], baseline: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Dict[str, List[str]]]:
    """Split a run into new, known, worsened and fixed scenarios.

    Args:
        results: The run.
        baseline: The committed baseline; loaded from :data:`BASELINE_PATH` when
            omitted.

    Returns:
        ``{"new": {...}, "known": {...}, "worsened": {...}, "fixed": {...}}``,
        each mapping a scenario name to the relevant finding codes.

        * ``new`` -- fails and is not in the baseline at all.  A regression.
        * ``worsened`` -- fails with at least one code the baseline did not
          record.  Also a regression, and easy to miss without this split.
        * ``known`` -- fails with exactly the codes the baseline expected.
        * ``fixed`` -- passes but is still listed.  The baseline must shrink.
    """
    base = load_baseline() if baseline is None else baseline
    out: Dict[str, Dict[str, List[str]]] = {
        "new": {}, "known": {}, "worsened": {}, "fixed": {}
    }
    for r in results:
        codes = sorted({f.code for f in r.findings})
        if r.passed:
            if r.scenario.name in base:
                out["fixed"][r.scenario.name] = base[r.scenario.name]
            continue
        if r.scenario.name not in base:
            out["new"][r.scenario.name] = codes
        elif set(codes) - set(base[r.scenario.name]):
            out["worsened"][r.scenario.name] = sorted(
                set(codes) - set(base[r.scenario.name])
            )
        else:
            out["known"][r.scenario.name] = codes
    return out


def all_scenarios() -> List[scen.Scenario]:
    """Every acceptance scenario, in report order.

    The library, followed by the independence cases in
    :data:`scenario.DEGRADED_STACK_SCENARIOS`.  Those live in ``scenario``
    because they are scenarios about the stack decomposition that module
    defines, but they are ordinary acceptance cases in every other respect and
    belong in the table, the JSON artifact and the baseline.
    """
    return list(SCENARIOS) + list(scen.DEGRADED_STACK_SCENARIOS)


def run_all(scenarios: Optional[Sequence[scen.Scenario]] = None) -> List[scen.ScenarioResult]:
    """Run every scenario and return the results in report order."""
    return [scen.run(s) for s in (scenarios if scenarios is not None else all_scenarios())]


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #


def _wrap(text: str, width: int, indent: str) -> str:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        lines.append(cur)
    return "\n".join(indent + ln for ln in lines)


def render_table(results: Sequence[scen.ScenarioResult], verbose: bool = False) -> str:
    """A pass/fail table with one diagnosis line per failure."""
    out: List[str] = []
    name_w = max([len(r.scenario.name) for r in results] + [8])
    out.append("=" * 100)
    out.append(
        "ADAS LONGITUDINAL SAFETY ACCEPTANCE  (spec %s, %d scenarios)"
        % (SPEC_VERSION, len(results))
    )
    out.append(
        "margins: contact = %.1f m, required clearance = %.1f m, comfort = %.1f m/s^2, "
        "emergency = %.1f m/s^2, authority = %.1f m/s^2"
        % (
            truth.CONTACT_GAP_M,
            truth.REQUIRED_CLEARANCE_M,
            truth.COMFORT_DECEL_MPS2,
            truth.EMERGENCY_DECEL_MPS2,
            scen.DEFAULT_PLANT.max_brake_decel_mps2,
        )
    )
    out.append("=" * 100)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        m = r.metrics
        out.append(
            "%-4s %-*s  min_gap=%-7s peak_brake=%-5s peak_jerk=%-7s pedal_conflict=%-3s "
            "worst_state=%s"
            % (
                status,
                name_w,
                r.scenario.name,
                _fmt(m["min_gap_m"]),
                _fmt(m["max_commanded_decel_mps2"]),
                _fmt(m["max_commanded_jerk_mps3"]),
                m["pedal_conflict_frames"],
                m["worst_safety_state"],
            )
        )
        if not r.scenario.stack.is_default:
            out.append("       stack:  %s" % r.scenario.stack.label)
        if m["contact"]:
            c = m["contact"]
            out.append(
                "       CONTACT at frame %d (t=%.2f s) with %r: closing %.2f m/s, "
                "ego %.2f m/s, gap %.2f m -- the run stops here"
                % (c["frame"], c["t_s"], c["label"], c["closing_mps"],
                   c["ego_v_mps"], c["gap_m"])
            )
        if not m["scenario_feasible"]:
            out.append(
                "       MIS-SPECIFIED: affordable decision latency %s s -- no system can "
                "pass this scenario as written"
                % _fmt(m["affordable_decision_latency_s"])
            )
        if m["self_reported_violations"]:
            out.append(
                "       system's own findings: %s"
                % ", ".join(
                    "%s x%d" % (k, v) for k, v in sorted(m["self_reported_violations"].items())
                )
            )
        if r.scenario.guards:
            out.append("       guards: %s" % r.scenario.guards)
        for f in r.findings:
            out.append("       %-22s %s" % (f.code.upper(), ""))
            out.append(_wrap(f.detail, 84, " " * 10))
        if verbose or not r.passed:
            out.append(_wrap("why: " + r.scenario.physics, 84, " " * 10))
        out.append("")
    passed = sum(1 for r in results if r.passed)
    out.append("-" * 100)
    out.append("%d passed, %d failed, %d total" % (passed, len(results) - passed, len(results)))

    split = classify_against_baseline(results)
    if load_baseline():
        out.append("")
        out.append(
            "against the committed baseline (%s): %d known, %d NEW, %d WORSENED, %d fixed"
            % (
                os.path.relpath(BASELINE_PATH),
                len(split["known"]),
                len(split["new"]),
                len(split["worsened"]),
                len(split["fixed"]),
            )
        )
        for name in sorted(split["new"]):
            out.append("  NEW       %-34s %s" % (name, ", ".join(split["new"][name])))
        for name in sorted(split["worsened"]):
            out.append("  WORSENED  %-34s %s" % (name, ", ".join(split["worsened"][name])))
        for name in sorted(split["fixed"]):
            out.append(
                "  FIXED     %-34s remove it from the baseline" % name
            )
        if not (split["new"] or split["worsened"] or split["fixed"]):
            out.append("  no change against the baseline; the known defects are still there")

    failures = [r for r in results if not r.passed]
    if failures:
        out.append("")
        out.append("diagnoses:")
        counts: Dict[str, int] = {}
        for r in failures:
            for f in r.findings:
                counts[f.code] = counts.get(f.code, 0) + 1
        for code in sorted(counts, key=lambda c: -counts[c]):
            out.append("  %-24s %d scenario(s)" % (code, counts[code]))
    out.append("-" * 100)
    return "\n".join(out)


def render_budgets(scenarios: Sequence[scen.Scenario]) -> str:
    """The affordable decision latency for every scenario, as a table.

    The single most useful number the redesign has.  It says, per scenario, how
    long the system may take to decide -- measured from the first frame on which
    the measurements it is handed could support the decision, and after the
    pipeline's own 55 ms of sense latency and the two frames a closing rate
    needs to exist have already been paid for.  A negative figure is a scenario
    that demands a reaction before the information exists; the only way to pass
    one is to brake on a prior instead of a measurement, which is the phantom
    braking this harness punishes elsewhere, so such a scenario is MIS-SPECIFIED
    and has to move or go.

    ``ideal`` is the clearance a zero-decision-latency system holds and ``best``
    the clearance a real one holds; both are measured through the plant with a
    jerk-limited brake demand, because an oracle that judged avoidability
    against a one-frame step to full authority would be requiring a manoeuvre
    the ``excess_jerk`` finding forbids.
    """
    rows = [scen.feasibility(s) for s in scenarios]
    out: List[str] = []
    out.append("=" * 108)
    out.append("AFFORDABLE DECISION LATENCY  (spec %s)" % SPEC_VERSION)
    out.append(
        "how long the system may take to decide, after the pipeline's own latency "
        "has already been paid"
    )
    out.append("=" * 108)
    name_w = max([len(s.name) for s in scenarios] + [8])
    out.append(
        "%-*s %9s %9s %9s %7s %9s %10s"
        % (name_w, "scenario", "req_clr", "ideal", "best", "act@", "min_lat", "AFFORD")
    )
    out.append("-" * 108)
    infeasible: List[str] = []
    for f in rows:
        if not f.feasible:
            infeasible.append(f.name)
        out.append(
            "%-*s %9s %9s %9s %7s %9s %10s%s"
            % (
                name_w,
                f.name,
                _fmt(f.required_clearance_m),
                _fmt(f.ideal_clearance_m),
                _fmt(f.best_clearance_m),
                f.earliest_actionable_frame
                if f.earliest_actionable_frame is not None
                else "-",
                _fmt(f.minimum_latency_s),
                _fmt(f.affordable_decision_latency_s),
                "   MIS-SPECIFIED" if not f.feasible else "",
            )
        )
    out.append("-" * 108)
    if infeasible:
        out.append(
            "%d scenario(s) cannot be passed by ANY system as written: %s"
            % (len(infeasible), ", ".join(infeasible))
        )
    else:
        out.append("every scenario has a non-negative decision budget")
    out.append("-" * 108)
    return "\n".join(out)


def render_reference(scenarios: Sequence[scen.Scenario]) -> str:
    """Run every scenario against :class:`scenario.ReferenceController`.

    THE SATISFIABILITY PROOF, and the redesign's known-achievable target.  The
    reference controller is a constant time gap, a braking law derived from
    required deceleration, and a jerk limiter; it pays the same sense latency
    and the same actuator lag as the system under test, and it never brakes on a
    closing rate it has not measured.  Every scenario must be passable by it.
    Anything it fails with a code outside
    :data:`scenario.SCENARIO_DEFECT_CODES` is a specification the harness cannot
    justify; anything it fails with a code inside that set is a scenario that
    has to be moved or deleted.
    """
    out: List[str] = []
    out.append("=" * 108)
    out.append("REFERENCE CONTROLLER  (spec %s) -- is this specification satisfiable?" % SPEC_VERSION)
    out.append("=" * 108)
    name_w = max([len(s.name) for s in scenarios] + [8])
    clean = 0
    spec_defects: Dict[str, List[str]] = {}
    real: Dict[str, List[str]] = {}
    for s in scenarios:
        result = scen.run(s, scen.reference_stack(s))
        codes = sorted({f.code for f in result.findings})
        system_codes = [c for c in codes if c not in scen.SCENARIO_DEFECT_CODES]
        m = result.metrics
        if not codes:
            tag = "PASS"
            clean += 1
        elif system_codes:
            tag = "UNSATISFIABLE"
            real[s.name] = codes
        else:
            tag = "SCENARIO-DEFECT"
            spec_defects[s.name] = codes
        out.append(
            "%-15s %-*s min_gap=%-8s peak_brake=%-6s peak_jerk=%-7s v_min=%-6s %s"
            % (
                tag,
                name_w,
                s.name,
                _fmt(m["min_gap_m"]),
                _fmt(m["max_commanded_decel_mps2"]),
                _fmt(m["max_commanded_jerk_mps3"]),
                _fmt(m["min_ego_speed_mps"]),
                ", ".join(codes),
            )
        )
    out.append("-" * 108)
    out.append(
        "%d of %d scenarios are satisfied by the reference controller; "
        "%d are mis-specified; %d are UNSATISFIABLE as written"
        % (clean, len(scenarios), len(spec_defects), len(real))
    )
    for name in sorted(spec_defects):
        out.append("  MIS-SPECIFIED  %-38s %s" % (name, ", ".join(spec_defects[name])))
    for name in sorted(real):
        out.append("  UNSATISFIABLE  %-38s %s" % (name, ", ".join(real[name])))
    out.append("-" * 108)
    return "\n".join(out)


def render_expectation_audit() -> str:
    """Which :class:`scenario.Expectation` fields ``evaluate`` actually reads."""
    reads = scen.expectation_field_reads()
    out = ["expectation fields and the number of read sites in evaluate():"]
    for name in sorted(reads):
        flag = "  <-- DECLARED BUT NEVER READ" if reads[name] == 0 else ""
        out.append("  %-38s %d%s" % (name, reads[name], flag))
    dead = scen.unread_expectation_fields()
    out.append(
        "no silently-ignored assertions"
        if not dead
        else "SILENTLY IGNORED: %s" % ", ".join(dead)
    )
    return "\n".join(out)


def _fmt(value: object) -> str:
    """Format a metric for the table, keeping the infinities distinguishable.

    ``inf`` and ``-inf`` used to print as the same dash, which made a scenario
    with an unbounded decision budget look identical to one that cannot be
    passed at all.
    """
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        return "%.2f" % value
    return str(value)


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


def to_json(results: Sequence[scen.ScenarioResult]) -> Dict[str, object]:
    """The machine-readable artifact.

    Structure::

        {"spec_version", "margins", "summary", "scenarios": [ ... ]}

    Each scenario entry carries its expectation, its physics justification, its
    metrics and the list of findings, so a regression can be diagnosed from the
    artifact alone.
    """
    scenarios = []
    for r in results:
        scenarios.append(
            {
                "name": r.scenario.name,
                "summary": r.scenario.summary,
                "guards": r.scenario.guards,
                "physics": r.scenario.physics,
                "frames": r.scenario.frames,
                "ego_speed_mps": r.scenario.ego_speed_mps,
                "stack": r.scenario.stack.label,
                "lead": r.scenario.lead.label if r.scenario.lead is not None else None,
                "road": r.scenario.road.label,
                "expectation": {
                    k: (v.value if hasattr(v, "value") else v)
                    for k, v in asdict(r.scenario.expect).items()
                    if v is not None and v is not False
                },
                "passed": r.passed,
                "findings": [{"code": f.code, "detail": f.detail} for f in r.findings],
                "scenario_defect": bool(r.findings) and all(
                    f.code in scen.SCENARIO_DEFECT_CODES for f in r.findings
                ),
                "feasibility": {
                    k: (v if not isinstance(v, float) else _round_json(v))
                    for k, v in asdict(scen.feasibility(r.scenario)).items()
                },
                "metrics": r.metrics,
            }
        )
    passed = sum(1 for r in results if r.passed)
    return {
        "spec_version": SPEC_VERSION,
        "margins": {
            "contact_gap_m": truth.CONTACT_GAP_M,
            "required_clearance_m": truth.REQUIRED_CLEARANCE_M,
            "comfort_decel_mps2": truth.COMFORT_DECEL_MPS2,
            "emergency_decel_mps2": truth.EMERGENCY_DECEL_MPS2,
            "negligible_decel_mps2": truth.NEGLIGIBLE_DECEL_MPS2,
            "headway_decel_allowance_mps2": truth.HEADWAY_DECEL_ALLOWANCE_MPS2,
            "comfort_jerk_mps3": truth.COMFORT_JERK_MPS3,
            "emergency_jerk_mps3": truth.EMERGENCY_JERK_MPS3,
            "pedal_conflict_eps": scen.PEDAL_CONFLICT_EPS,
            "justification_tolerance_mps2": truth.JUSTIFICATION_TOLERANCE_MPS2,
            "justification_window_frames": truth.JUSTIFICATION_WINDOW_FRAMES,
            "max_brake_authority_mps2": scen.DEFAULT_PLANT.max_brake_decel_mps2,
            "dt_s": scen.DEFAULT_PLANT.dt_s,
            "sense_latency_s": scen.PERFECT_PERCEPTION.sense_latency_s,
            "brake_rise_time_s": scen.DEFAULT_PLANT.brake_rise_time_s,
            "actuation_latency_s": scen.DEFAULT_PLANT.actuation_latency_s,
        },
        "expectation_field_reads": scen.expectation_field_reads(),
        "unread_expectation_fields": scen.unread_expectation_fields(),
        "mis_specified": [
            r.scenario.name
            for r in results
            if not scen.feasibility(r.scenario).feasible
        ],
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "failing": [r.scenario.name for r in results if not r.passed],
        },
        "baseline": classify_against_baseline(results),
        "scenarios": scenarios,
    }


def _round_json(x: float) -> object:
    """Floats for the artifact: rounded, with the infinities left readable."""
    if x != x or x in (float("inf"), float("-inf")):
        return str(x)
    return round(float(x), 4)


def write_json(path: str, results: Sequence[scen.ScenarioResult]) -> None:
    """Write the JSON artifact, sorted and indented so diffs are readable."""
    with open(path, "w") as handle:
        json.dump(to_json(results), handle, indent=2, sort_keys=True)
        handle.write("\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the library, print the table, optionally write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", metavar="PATH", help="also write the JSON artifact here")
    parser.add_argument("--verbose", action="store_true", help="print the physics for every case")
    parser.add_argument("--only", metavar="NAME", action="append", help="run only these scenarios")
    parser.add_argument(
        "--budgets",
        action="store_true",
        help="print the affordable decision latency for every scenario and exit; "
        "a negative figure is a scenario no system can pass",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="run the reference controller instead of the system under test and "
        "print whether this specification is satisfiable at all",
    )
    parser.add_argument(
        "--audit-expectations",
        action="store_true",
        help="print every Expectation field and how many times evaluate() reads it",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite %s from this run; refuses a partial run, because a baseline "
        "built from a subset would silently mark every unrun scenario as fixed"
        % os.path.relpath(BASELINE_PATH),
    )
    parser.add_argument(
        "--log-level",
        default="ERROR",
        help="level for the adas.* loggers; the stack logs a WARNING for every "
        "arbitration it changes, which is thousands of lines over the library "
        "(default: ERROR)",
    )
    args = parser.parse_args(argv)
    logging.getLogger("adas").setLevel(getattr(logging, args.log_level.upper(), logging.ERROR))

    chosen = all_scenarios()
    if args.only:
        wanted = set(args.only)
        chosen = [s for s in chosen if s.name in wanted]
        missing = wanted - {s.name for s in chosen}
        if missing:
            parser.error("unknown scenario(s): %s" % ", ".join(sorted(missing)))

    if args.audit_expectations:
        print(render_expectation_audit())
        return 0 if not scen.unread_expectation_fields() else 3

    if args.budgets:
        print(render_budgets(chosen))
        return 0 if all(scen.feasibility(s).feasible for s in chosen) else 3

    if args.reference:
        print(render_reference(chosen))
        bad = 0
        for s in chosen:
            codes = {f.code for f in scen.run(s, scen.reference_stack(s)).findings}
            if codes - set(scen.SCENARIO_DEFECT_CODES):
                bad += 1
            elif codes:
                bad += 1
        return 0 if bad == 0 else 3

    results = run_all(chosen)
    print(render_table(results, verbose=args.verbose))
    if args.json:
        write_json(args.json, results)
        print("wrote %s" % args.json)
    if args.write_baseline:
        if args.only:
            parser.error("--write-baseline needs the whole library, not --only")
        write_baseline(results)
        print("wrote %s (%d known failures)" % (BASELINE_PATH, sum(1 for r in results if not r.passed)))
    split = classify_against_baseline(results)
    if split["new"] or split["worsened"]:
        return 2
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
