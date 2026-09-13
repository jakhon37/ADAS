#!/usr/bin/env python3
"""Record a real ADAS run, and analyse every arbiter intervention in it.

Two subcommands, deliberately split so the expensive half runs once:

``record``   drives the real pipeline (YOLOX + UFLD over a video) with the
             arbiter instrumented, and writes a JSON recording.  **This is the
             only part that needs the GPU**, so it must run under the board's
             GPU mutex::

                 ssh jetson-nx 'flock /tmp/jetson-gpu.lock -c "cd ~/myspace/ADAS \\
                     && python3 scripts/analyze_run.py record --out /tmp/run400.json"'

``analyze``  reads that file and prints the justification report.  No GPU, no
             TensorRT, no camera; runs anywhere, as often as you like.

The analysis itself lives in :mod:`tests.scenarios.footage`; this is the command
line around it.  Read the report from the INTERVENTIONS section down: every
frame that intervened is judged against the range history the arbiter was
actually handed, and every frame that did *not* intervene is checked for a
hazard it should have reacted to.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(_REPO_ROOT, "src"), _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tests.scenarios.footage import (  # noqa: E402
    AnalysisParams,
    RunRecording,
    analyse,
    format_report,
    record_run,
)
from tests.scenarios.sweep import SweepSpec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="drive a live instrumented run (needs the GPU)")
    rec.add_argument("--out", required=True, help="where to write the recording")
    rec.add_argument(
        "--source",
        default="Ultra-Fast-Lane-Detection-v2/example.mp4",
        help="video path, relative to the repo root",
    )
    rec.add_argument("--frames", type=int, default=400, help="frames to run (default 400)")
    rec.add_argument("--ego-speed", type=float, default=15.0, help="simulated ego speed, m/s")
    rec.add_argument("--detector", default="yolox")
    rec.add_argument("--lane", default="ufld")
    rec.add_argument("--target-fps", type=float, default=20.0)
    rec.add_argument("--config", default=None, help="optional JSON config")
    rec.add_argument("--analyze", action="store_true", help="also print the report")
    rec.add_argument(
        "--quiet-pipeline",
        action="store_true",
        help="silence the pipeline's own INFO/WARNING logging during the run",
    )

    ana = sub.add_parser("analyze", help="analyse a recording (no GPU)")
    ana.add_argument("recording", help="a file written by `record`")
    ana.add_argument(
        "--brake-threshold",
        type=float,
        default=0.10,
        help="a brake above this counts as an intervention (default 0.10)",
    )
    ana.add_argument(
        "--hard-brake-threshold",
        type=float,
        default=0.50,
        help="a brake above this is a full-authority intervention (default 0.50)",
    )
    ana.add_argument(
        "--window",
        type=int,
        default=12,
        help="frames of raw range history per intervention (default 12)",
    )
    ana.add_argument("--comfort-decel", type=float, default=None, help="override, m/s^2")
    ana.add_argument("--reaction", type=float, default=None, help="override, s")
    ana.add_argument("--max-listed", type=int, default=60, help="cap the per-frame listing")
    ana.add_argument("--all", action="store_true", help="list every intervention, not just the unjustified")
    ana.add_argument("--json", dest="json_path", help="write the report as JSON")
    ana.add_argument(
        "--fail-on-unjustified",
        action="store_true",
        help="exit non-zero when any intervention is unjustified",
    )
    ana.add_argument(
        "--fail-on-missed",
        action="store_true",
        help="exit non-zero when any measured hazard got no reaction",
    )
    return parser


def _params(args: argparse.Namespace) -> AnalysisParams:
    """Build the analysis parameters from the command line."""
    spec_fields = {}
    if args.comfort_decel is not None:
        spec_fields["comfort_decel_mps2"] = args.comfort_decel
    if args.reaction is not None:
        spec_fields["reaction_s"] = args.reaction
    return AnalysisParams(
        spec=SweepSpec(**spec_fields),
        brake_threshold=args.brake_threshold,
        hard_brake_threshold=args.hard_brake_threshold,
        history_frames=args.window,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.  Returns the process exit status."""
    args = build_parser().parse_args(argv)

    if args.command == "record":
        if args.quiet_pipeline:
            logging.disable(logging.WARNING)
        recording = record_run(
            source=args.source,
            frames=args.frames,
            ego_speed_mps=args.ego_speed,
            detector=args.detector,
            lane=args.lane,
            target_fps=args.target_fps,
            config_path=args.config,
        )
        recording.save(args.out)
        print(
            "recorded %d arbitrated frames -> %s"
            % (len(recording.frames), args.out)
        )
        if recording.meta.get("missing_hooks"):
            print(
                "WARNING: these instrument hooks did not attach: %s"
                % ", ".join(recording.meta["missing_hooks"])
            )
        if args.analyze:
            report = analyse(recording)
            print(format_report(report))
        return 0

    recording = RunRecording.load(args.recording)
    report = analyse(recording, _params(args))
    print(format_report(report, max_interventions=args.max_listed, show_all=args.all))
    if args.json_path:
        with open(args.json_path, "w") as handle:
            json.dump(report.to_dict(), handle, indent=1, sort_keys=True)
        print()
        print("wrote %s" % args.json_path)

    status = 0
    if args.fail_on_unjustified and report.unjustified:
        print()
        print("FAIL: %d unjustified interventions" % len(report.unjustified))
        status = 1
    if args.fail_on_missed and report.missed:
        print("FAIL: %d missed reactions" % len(report.missed))
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
