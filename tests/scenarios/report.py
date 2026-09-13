"""Reporting: a readable pass/fail table and a machine-readable JSON artifact.

Run the whole library and print the table::

    python -m tests.scenarios.report

Write the JSON artifact as well::

    python -m tests.scenarios.report --json build/safety_acceptance.json

Show the physics justification for every scenario, passing or failing::

    python -m tests.scenarios.report --verbose

The table is the human deliverable and the JSON is the machine one; both carry
the same diagnoses, so a CI job can diff two runs and a person can read one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

from tests.scenarios import oracle as truth
from tests.scenarios import scenario as scen
from tests.scenarios.library import SCENARIOS

SPEC_VERSION = "1.0"
"""Bumped whenever a margin convention or an authority limit changes.

A JSON artifact from a different spec version is not comparable with this one.
"""


def run_all(scenarios: Optional[Sequence[scen.Scenario]] = None) -> List[scen.ScenarioResult]:
    """Run every scenario and return the results in library order."""
    return [scen.run(s) for s in (scenarios if scenarios is not None else SCENARIOS)]


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
            "%-4s %-*s  min_gap=%-7s peak_brake=%-5s worst_state=%s"
            % (
                status,
                name_w,
                r.scenario.name,
                _fmt(m["min_gap_m"]),
                _fmt(m["max_commanded_decel_mps2"]),
                m["worst_safety_state"],
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


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return "-"
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
                "lead": r.scenario.lead.label if r.scenario.lead is not None else None,
                "road": r.scenario.road.label,
                "expectation": {
                    k: (v.value if hasattr(v, "value") else v)
                    for k, v in asdict(r.scenario.expect).items()
                    if v is not None and v is not False
                },
                "passed": r.passed,
                "findings": [{"code": f.code, "detail": f.detail} for f in r.findings],
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
            "justification_tolerance_mps2": truth.JUSTIFICATION_TOLERANCE_MPS2,
            "justification_window_frames": truth.JUSTIFICATION_WINDOW_FRAMES,
            "max_brake_authority_mps2": scen.DEFAULT_PLANT.max_brake_decel_mps2,
            "dt_s": scen.DEFAULT_PLANT.dt_s,
        },
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "failing": [r.scenario.name for r in results if not r.passed],
        },
        "scenarios": scenarios,
    }


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
        "--log-level",
        default="ERROR",
        help="level for the adas.* loggers; the stack logs a WARNING for every "
        "arbitration it changes, which is thousands of lines over the library "
        "(default: ERROR)",
    )
    args = parser.parse_args(argv)
    logging.getLogger("adas").setLevel(getattr(logging, args.log_level.upper(), logging.ERROR))

    chosen = SCENARIOS
    if args.only:
        wanted = set(args.only)
        chosen = [s for s in SCENARIOS if s.name in wanted]
        missing = wanted - {s.name for s in chosen}
        if missing:
            parser.error("unknown scenario(s): %s" % ", ".join(sorted(missing)))

    results = run_all(chosen)
    print(render_table(results, verbose=args.verbose))
    if args.json:
        write_json(args.json, results)
        print("wrote %s" % args.json)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
