#!/usr/bin/env python3
"""Run the longitudinal safety envelope sweep and report the failure REGIONS.

THIS IS THE PRIMARY LONGITUDINAL SAFETY GATE.  The hand-picked scenario library
in :mod:`tests.scenarios.library` is the readable specification, but it missed
both of this module's historical failures and the sweep found both, for the
reason a sweep always beats a sample: the useful output of a sweep is not a pass
count but a BOUNDARY.  "It brakes for any in-path track closer than 43 m at
20 m/s" is a diagnosis; "312/336 passed" is not.

The sweep and its physics live in :mod:`tests.scenarios.sweep`; this is the
command line around them.  Nothing here needs a GPU, TensorRT, a camera or the
network, and the whole run is deterministic: the same arguments produce the same
bytes, so two runs can be diffed and the JSON artifact can be committed.

Three gradings are reported, independently, because a cell can be right about
one and wrong about another:

``verdict``         emergency authority: COLLISION / MISSED / PHANTOM / LATE /
                    EARLY / CORRECT.
``headway_verdict`` the soft response against the RSS-style safe gap.
``band_verdict``    the SUB-EMERGENCY band, 3.0 to 3.5 m/s^2 -- braking that is
                    beyond comfort but below the emergency threshold, and
                    therefore invisible to both of the other two.

Examples::

    # the CI gate: ~16 s, 120 cells straddling every measured boundary,
    # non-zero exit on any collision, miss, phantom, lateness or band fault
    python3 scripts/run_safety_sweep.py --gate

    # the default envelope, full report and a machine-readable dump
    python3 scripts/run_safety_sweep.py --json build/safety_sweep.json

    # hunt a boundary by hand at whatever resolution you like
    python3 scripts/run_safety_sweep.py --ego 15 \\
        --range 18,20,22,24,26,28,30,32 --rate 0 --lead-decel 0

Read the output from the bottom up: the REGIONS sections are the useful part.  A
count tells you how bad it is; a region tells you what is wrong.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The repository root goes FIRST, because ``tests.scenarios`` -- the
# specification -- must always be this working tree's.  The working tree's
# ``src`` goes LAST, so that an explicit ``PYTHONPATH`` naming another
# checkout's ``src`` wins and ``import adas`` resolves to the system under
# test the caller asked for.
#
# It used to be the other way round, and that was a defect in the gate: an
# earlier prepend of ``<repo>/src`` shadowed every PYTHONPATH entry, so
# backtesting an older commit with
# ``PYTHONPATH=<worktree>/src:. python3 scripts/run_safety_sweep.py --gate``
# silently graded the WORKING TREE and printed a confident, wrong table.
# Measured: the phantom commit 25e3ba5 and the missed-braking commit 1ce4886
# both reported the identical failure line, which is impossible for two
# arbiters with opposite defects.  ``--provenance`` below now prints which
# arbiter actually loaded, so the same mistake cannot be silent again.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from tests.scenarios.plant import DEFAULT_PLANT  # noqa: E402
from tests.scenarios.sweep import (  # noqa: E402
    BAND_CHARS,
    BAND_FAILURES,
    PROFILES,
    SweepGrid,
    SweepSpec,
    Verdict,
    render_grids,
    run_sweep,
    summarise,
)

GATE_FAIL_ON = "COLLISION,MISSED,PHANTOM,LATE,EARLY,BAND_UNWARRANTED"
"""The verdicts that fail the committed gate.

Every one of them is a defect and none of them is a matter of taste:
``COLLISION`` and ``MISSED`` are the passive direction, ``PHANTOM`` and
``EARLY`` the aggressive one, ``LATE`` is intervening after the last moment the
vehicle could still stop with room, and ``BAND_UNWARRANTED`` is sustained
braking above comfort with neither a hazard nor a headway deficit to correct.
A gate that omitted either direction is how this arbiter came to oscillate
between them.
"""

GATE_PROFILE = "fast"
"""Grid the gate runs.  120 cells, about sixteen seconds, and it holds a
straddling pair of ranges either side of every boundary the envelope is known to
contain -- which is what makes a sixteen-second grid worth running instead of a
fifteen-hundred cell one.  The two ranges beyond 70 m were added because the
tool reported two failure regions running off the top of the old axis; see
``AXIS COVERAGE`` in the output and ``PROFILES`` in tests/scenarios/sweep.py."""


def _floats(text: str) -> tuple:
    """``"5, 10,15"`` -> ``(5.0, 10.0, 15.0)``."""
    return tuple(float(part) for part in text.replace(" ", "").split(",") if part)


def build_parser() -> argparse.ArgumentParser:
    """Command line."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        default="standard",
        choices=sorted(PROFILES),
        help="grid resolution (default: standard)",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="run as the committed CI gate: the '%s' profile, no progress line, "
        "and a non-zero exit on %s. Equivalent to "
        "--profile %s --quiet --fail-on %s, and overrides --profile and "
        "--fail-on so that the gate cannot be quietly weakened by a flag."
        % (GATE_PROFILE, GATE_FAIL_ON, GATE_PROFILE, GATE_FAIL_ON),
    )
    axes = parser.add_argument_group(
        "custom axes",
        "Any axis given here overrides the profile's. Comma-separated floats.",
    )
    axes.add_argument("--ego", type=_floats, help="ego speeds, m/s")
    axes.add_argument("--range", dest="ranges", type=_floats, help="initial ranges, m")
    axes.add_argument("--rate", dest="rates", type=_floats, help="initial closing rates, m/s (negative closes)")
    axes.add_argument("--lead-decel", dest="decels", type=_floats, help="lead decelerations, m/s^2")

    spec = parser.add_argument_group("specification", "Physics and requirement constants.")
    spec.add_argument("--dt", type=float, default=None, help="frame interval, s (default 0.05)")
    spec.add_argument("--horizon", type=float, default=None, help="arbitration episode length, s")
    spec.add_argument("--reaction", type=float, default=None, help="system reaction latency, s")
    spec.add_argument("--comfort-decel", type=float, default=None, help="comfort deceleration, m/s^2")

    out = parser.add_argument_group("output")
    out.add_argument("--json", dest="json_path", help="write the full result set here")
    out.add_argument("--no-grid", action="store_true", help="skip the ASCII grids")
    out.add_argument("--no-cells", action="store_true", help="skip the per-cell failure listing")
    out.add_argument("--max-cells-listed", type=int, default=40, help="cap the failure listing")
    out.add_argument("--no-instrument", action="store_true", help="do not capture the arbiter's own demand")
    out.add_argument("--quiet", action="store_true", help="no progress line")
    out.add_argument(
        "--verbose-arbiter",
        action="store_true",
        help="let the arbiter's own WARNING logging through. Off by default: the "
        "sweep provokes tens of thousands of interventions on purpose and the log "
        "would bury the report.",
    )
    out.add_argument(
        "--fail-on",
        default="",
        help="comma-separated verdicts that make this exit non-zero, "
        "e.g. COLLISION,MISSED,PHANTOM. Empty means always exit 0.",
    )
    return parser


def _grid_from_args(args: argparse.Namespace) -> SweepGrid:
    """Profile, with any explicitly given axis substituted in."""
    base = PROFILES[args.profile]
    return SweepGrid(
        name=base.name if not any((args.ego, args.ranges, args.rates, args.decels)) else "custom",
        ego_speeds_mps=args.ego or base.ego_speeds_mps,
        ranges_m=args.ranges or base.ranges_m,
        relative_rates_mps=args.rates or base.relative_rates_mps,
        lead_decels_mps2=args.decels or base.lead_decels_mps2,
        horizon_s=args.horizon if args.horizon is not None else base.horizon_s,
    )


def _spec_from_args(args: argparse.Namespace) -> SweepSpec:
    """Defaults, with any overridden constant substituted in."""
    fields = {}
    if args.dt is not None:
        fields["dt_s"] = args.dt
    if args.horizon is not None:
        fields["horizon_s"] = args.horizon
    if args.reaction is not None:
        fields["reaction_s"] = args.reaction
    if args.comfort_decel is not None:
        fields["comfort_decel_mps2"] = args.comfort_decel
    return SweepSpec(**fields)


def _arbiter_provenance() -> tuple:
    """``(path, md5)`` of the ``adas.control.arbiter`` module that actually loaded.

    Printed in the header of every run.  A sweep is only evidence about the code
    it graded, and "which code was that" is the one question a table of numbers
    cannot answer for itself.  It is also what makes backtesting an old commit
    checkable at a glance: the md5 in the header must be that commit's arbiter
    md5, and if it is the working tree's then the run graded HEAD whatever
    ``PYTHONPATH`` said.

    Any failure to resolve the module is reported in place rather than raised:
    the sweep's job is to grade the arbiter, not to police its own header.
    """
    try:
        import hashlib
        import adas.control.arbiter as _arb

        path = (_arb.__file__ or "").replace(".pyc", ".py")
        with open(path, "rb") as handle:
            return path, hashlib.md5(handle.read()).hexdigest()[:8]
    except Exception as exc:  # pragma: no cover - defensive header only
        return "<unresolved: %s>" % exc, "?"


def _print_axis_coverage(report: dict, grid: SweepGrid) -> None:
    """State, in one place, whether every failure region is bounded on both sides.

    A region that runs to the top of the range axis has an unknown width: the
    sweep has measured that the failure occurs at 70 m and nothing at all about
    where it stops, so "EARLY for range 16-70" is a lower bound wearing the
    costume of a measurement.  The individual region statements already say so,
    but they say it in the middle of forty other lines and it was missed for a
    whole revision.  This prints the count and names the offenders, so extending
    the axis is a decision somebody takes rather than one nobody notices.

    It is deliberately NOT part of ``--fail-on``: an unbounded region is a
    deficiency in the GRID, not a defect in the arbiter, and conflating the two
    would let a harness bug read as a system failure -- which is the whole
    disease this workstream exists to treat.
    """
    open_ended = []
    for source in (report["regions"], report["band_regions"]):
        for verdict in sorted(source):
            for row in source[verdict]:
                if row["open_ended"]:
                    open_ended.append((verdict, row))
    print()
    print("-" * 78)
    print("AXIS COVERAGE -- is every failure region bounded on both sides?")
    print("-" * 78)
    if not open_ended:
        print(
            "  yes: every failure region closes inside the swept range axis "
            "(top = %g m), so every reported width is a measurement."
            % max(grid.ranges_m)
        )
        return
    print(
        "  NO -- %d region(s) run off the top of the range axis (%g m). Their widths "
        "are LOWER BOUNDS, not measurements. Extend --range past the top, re-measure, "
        "and move the profile's axis to straddle whatever boundary you find."
        % (len(open_ended), max(grid.ranges_m))
    )
    for verdict, row in open_ended:
        print(
            "    %-16s ego %g m/s, rate %+g, lead_decel %g: %s m and still failing"
            % (
                verdict,
                row["ego_speed_mps"],
                row["relative_rate_mps"],
                row["lead_decel_mps2"],
                row["range_span"],
            )
        )


def _progress(total: int):
    """Single rewritten progress line on a tty, nothing on a pipe."""
    if not sys.stderr.isatty():
        return None

    def tick(done: int, _total: int) -> None:
        sys.stderr.write("\r  sweeping %d/%d cells" % (done, total))
        if done == total:
            sys.stderr.write("\n")
        sys.stderr.flush()

    return tick


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.  Returns the process exit status."""
    args = build_parser().parse_args(argv)
    if args.gate:
        args.profile = GATE_PROFILE
        args.fail_on = GATE_FAIL_ON
        args.quiet = True
    if not args.verbose_arbiter:
        logging.disable(logging.WARNING)
    grid = _grid_from_args(args)
    spec = _spec_from_args(args)
    cells = grid.cells()

    print("=" * 78)
    print("LONGITUDINAL SAFETY ENVELOPE SWEEP")
    print("=" * 78)
    print("profile            : %s" % grid.name)
    print("ego speeds   (m/s) : %s" % ", ".join("%g" % v for v in sorted(grid.ego_speeds_mps)))
    print("ranges         (m) : %s" % ", ".join("%g" % v for v in sorted(grid.ranges_m)))
    print("closing rates(m/s) : %s" % ", ".join("%+g" % v for v in sorted(grid.relative_rates_mps)))
    print("lead decel (m/s^2) : %s" % ", ".join("%g" % v for v in sorted(grid.lead_decels_mps2)))
    print("cells              : %d" % len(cells))
    horizon = grid.horizon_s if grid.horizon_s is not None else spec.horizon_s
    print(
        "spec               : clearance %.1f m, comfort %.1f m/s^2, emergency %.1f "
        "m/s^2, authority %.1f m/s^2, headway reaction %.2f s"
        % (
            spec.standstill_gap_m,
            spec.comfort_decel_mps2,
            spec.emergency_decel_mps2,
            spec.brake_authority_mps2,
            spec.reaction_s,
        )
    )
    print(
        "episode            : dt %.3f s, arbitration horizon %.1f s, oracle horizon %.0f s"
        % (spec.dt_s, horizon, spec.oracle_horizon_s)
    )
    print(
        "incoming command   : throttle 0.40, brake 0.00 -- every brake below is the "
        "arbiter's own"
    )
    print(
        "plant / oracle     : tests.scenarios.plant (brake rise %.2f s) and "
        "tests.scenarios.oracle, shared with the scenario suite"
        % DEFAULT_PLANT.brake_rise_time_s
    )
    arbiter_path, arbiter_md5 = _arbiter_provenance()
    print("arbiter under test : %s  md5 %s" % (arbiter_path, arbiter_md5))
    print()

    results = run_sweep(
        grid,
        spec=spec,
        use_instrument=not args.no_instrument,
        progress=None if args.quiet else _progress(len(cells)),
    )
    report = summarise(results, grid, spec)

    print("-" * 78)
    print("COUNTS")
    print("-" * 78)
    total_graded = report["cells_graded"]
    for verdict in Verdict.ORDER:
        n = report["counts"].get(verdict, 0)
        if not n:
            continue
        share = (100.0 * n / total_graded) if (total_graded and verdict != Verdict.INFEASIBLE) else 0.0
        suffix = "" if verdict == Verdict.INFEASIBLE else "  (%.1f%% of graded)" % share
        print("  %-10s %4d%s" % (verdict, n, suffix))
    print("  %-10s %4d" % ("FAILURES", report["failures"]))
    print("  headway (soft response, graded separately): %s" % report["headway_counts"])
    print(
        "  sub-emergency band %.1f-%.1f m/s^2 (graded separately): %s"
        % (spec.comfort_decel_mps2, spec.emergency_decel_mps2, report["band_counts"])
    )
    if report["unwarranted_on_inferred_rate"]:
        print(
            "  of the PHANTOM and EARLY cells, %d fired on a frame where the closing "
            "rate was still the seeded prior, not a measurement"
            % report["unwarranted_on_inferred_rate"]
        )

    if not args.no_grid:
        print()
        print("-" * 78)
        print("GRIDS -- failure REGIONS are contiguous blocks of one letter")
        print("-" * 78)
        print(render_grids(results, grid))
        if report["band_failures"]:
            print()
            print("  SUB-EMERGENCY BAND -- B = commanded %.1f-%.1f m/s^2 with neither a "
                  "hazard nor an unsafe headway" % (spec.comfort_decel_mps2,
                                                    spec.emergency_decel_mps2))
            print(render_grids(results, grid, "band_verdict", BAND_CHARS))

    print()
    print("-" * 78)
    print("REGIONS -- the boundary of each failure region")
    print("-" * 78)
    regions = report["regions"]
    if not regions:
        print("  none")
    for verdict in Verdict.FAILURES:
        rows = regions.get(verdict) or []
        if not rows:
            continue
        print()
        print("  %s (%d slices)" % (verdict, len(rows)))
        for row in rows:
            print("    " + row["statement"])

    band_regions = report["band_regions"]
    print()
    print("-" * 78)
    print(
        "SUB-EMERGENCY BAND REGIONS -- %.1f to %.1f m/s^2, below the emergency "
        "threshold and above comfort" % (spec.comfort_decel_mps2, spec.emergency_decel_mps2)
    )
    print("-" * 78)
    if not any(band_regions.get(v) for v in BAND_FAILURES):
        print("  none")
    for verdict in BAND_FAILURES:
        for row in band_regions.get(verdict) or []:
            print("    " + row["statement"])
    for res in report["worst_band"][:5]:
        print(
            "      peak %.2f m/s^2 held for %d frame(s) from frame %s at %s"
            % (
                res["max_band_decel_mps2"],
                res["band_frames"],
                res["first_band_frame"],
                "ego %g m/s, range %g m, rate %+g, lead_decel %g"
                % (
                    res["ego_speed_mps"],
                    res["range_m"],
                    res["relative_rate_mps"],
                    res["lead_decel_mps2"],
                ),
            )
        )

    _print_axis_coverage(report, grid)

    if not args.no_cells:
        failing = [r for r in results if r.verdict in Verdict.FAILURES]
        print()
        print("-" * 78)
        print("FAILING CELLS (%d; showing up to %d)" % (len(failing), args.max_cells_listed))
        print("-" * 78)
        for res in failing[: args.max_cells_listed]:
            oracle = "warrant=%s mandate=%s lost=%s" % (
                res.warrant_frame,
                res.mandate_frame,
                res.lost_frame,
            )
            print(
                "  %-10s %s | %s | hard=%s soft=%s max_cmd_decel=%.2f max_demand=%.2f"
                % (
                    res.verdict,
                    res.cell.label(),
                    oracle,
                    res.hard_frame,
                    res.soft_frame,
                    res.max_decel_open_mps2,
                    res.max_demand_open_mps2,
                )
            )
            detail: List[str] = []
            if res.collided:
                detail.append("closed loop min gap %.2f m" % res.min_gap_m)
            if res.first_hard_state:
                detail.append("first hard state %s" % res.first_hard_state)
            if res.hard_with_inferred_rate:
                detail.append("rate was INFERRED, not measured")
            if res.first_hard_findings:
                detail.append("findings " + ",".join(res.first_hard_findings)[:90])
            for note in res.notes:
                detail.append(note)
            if detail:
                print("             " + "; ".join(detail))

    if args.json_path:
        payload = {
            "summary": report,
            "grid": {
                "name": grid.name,
                "ego_speeds_mps": list(grid.ego_speeds_mps),
                "ranges_m": list(grid.ranges_m),
                "relative_rates_mps": list(grid.relative_rates_mps),
                "lead_decels_mps2": list(grid.lead_decels_mps2),
                "horizon_s": grid.horizon_s,
            },
            "cells": [r.to_dict() for r in results],
        }
        with open(args.json_path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print()
        print("wrote %s" % args.json_path)

    counts_all = dict(report["counts"])
    counts_all.update(report["band_counts"])
    fail_on = {v.strip().upper() for v in args.fail_on.split(",") if v.strip()}
    if fail_on:
        hit = sorted(v for v in fail_on if counts_all.get(v, 0))
        if hit:
            print()
            print("FAIL: %s" % ", ".join("%s=%d" % (v, counts_all[v]) for v in hit))
            return 1
        print()
        print("OK: none of %s occurred" % ", ".join(sorted(fail_on)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
