#!/usr/bin/env python3
"""Fetch the YOLO ONNX and build its TensorRT engine when the Jetson is idle.

Compatibility shim. The real work now lives in the general model builder:

* ``scripts/fetch_models.sh``  - SHA-256-pinned downloads driven by
  ``models/MANIFEST.json``
* ``scripts/build_engines.py`` - ONNX -> TensorRT for every ADAS model, with
  the shared ``/tmp/jetson-gpu.lock`` mutex, engine verification and manifest
  bookkeeping

This entry point is kept because ``docs/JETSON.md``, ``README.md`` and existing
operator muscle memory reference it. It adds one thing the general builder
deliberately does not do: an optional *wait for the board to go idle* before
starting, which is useful when a replay session or the neighbouring DMS project
still owns the GPU.

Note on licensing: ``yolov5n`` is **AGPL-3.0**. ``--model yolox_nano`` builds
the Apache-2.0 replacement instead and is the right default for any build that
leaves this board. See ``models/MANIFEST.json`` for the full licence record.

Units: ``--min-avail-mb`` is MiB of ``MemAvailable``; ``--wait-timeout-s`` is
seconds.

Failure behaviour: exits non-zero if the idle wait times out, if the fetch
fails its hash check, or if ``trtexec`` fails. Nothing is written to
``models/`` on a failed build beyond the ``.build.log``, and the failure is
recorded under ``blocked`` in ``models/MANIFEST.json``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FETCH = ROOT / "scripts" / "fetch_models.sh"
BUILDER = ROOT / "scripts" / "build_engines.py"

#: Command-line fragments that mean somebody else is using the GPU or RAM.
BUSY_CMDS = ("dms.app", "trtexec", "adas.cli", "train.py")


def available_mb() -> int:
    """``MemAvailable`` in MiB, or 0 when /proc/meminfo cannot be read."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError as exc:
        print("cannot read /proc/meminfo (%s); skipping the RAM check" % exc)
        return 0
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def gpu_busy() -> bool:
    """True when tegrastats reports >=20 % 3D-engine utilisation.

    Returns False when ``tegrastats`` is missing or produces no GR3D sample --
    an unknown GPU state must not block the build forever, and the
    ``/tmp/jetson-gpu.lock`` mutex inside the builder is the real guard.
    """
    try:
        out = subprocess.check_output(
            ["timeout", "2", "tegrastats", "--interval", "500"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False
    lines = [ln for ln in out.strip().splitlines() if "GR3D_FREQ" in ln]
    if not lines:
        return False
    m = re.search(r"GR3D_FREQ\s+(\d+)%", lines[-1])
    return bool(m) and int(m.group(1)) >= 20


def other_heavy_process() -> str:
    """Return the command line of a neighbouring GPU/RAM hog, or ''."""
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except (subprocess.CalledProcessError, OSError):
        return ""
    me = str(os.getpid())
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2 or parts[0] == me:
            continue
        cmd = parts[1]
        if "build_yolo_engine" in cmd or "build_engines.py" in cmd:
            continue
        for needle in BUSY_CMDS:
            if needle in cmd:
                return cmd.strip()
    return ""


def wait_for_idle(min_avail_mb: int, timeout_s: int) -> None:
    """Block until RAM is free and nothing else owns the GPU.

    Raises ``SystemExit`` after *timeout_s* seconds rather than starting a
    build that would OOM the board.
    """
    deadline = time.time() + timeout_s
    while True:
        avail = available_mb()
        busy = other_heavy_process()
        gpu = gpu_busy()
        if avail >= min_avail_mb and not busy and not gpu:
            print("idle: avail=%s MiB, gpu idle, no neighbours" % avail)
            return
        if time.time() >= deadline:
            raise SystemExit(
                "timed out waiting for an idle board (avail=%s MiB busy=%r gpu_busy=%s)"
                % (avail, busy, gpu)
            )
        print("waiting: avail=%s MiB (need %s) busy=%s gpu_busy=%s"
              % (avail, min_avail_mb, busy or "-", gpu))
        time.sleep(15)


def run(cmd, **kwargs) -> int:
    print("+ %s" % " ".join(str(c) for c in cmd))
    sys.stdout.flush()
    return subprocess.call([str(c) for c in cmd], **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", default="yolov5n",
        help="model key in models/MANIFEST.json (default yolov5n, AGPL-3.0; "
             "prefer yolox_nano for Apache-2.0)",
    )
    parser.add_argument("--min-avail-mb", type=int, default=2500,
                        help="MemAvailable floor in MiB before building")
    parser.add_argument("--wait-timeout-s", type=int, default=3600)
    parser.add_argument("--no-wait", action="store_true",
                        help="build immediately; the GPU mutex still applies")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the engine already exists")
    # Accepted for backwards compatibility with the pre-manifest interface.
    parser.add_argument("--onnx", help=argparse.SUPPRESS)
    parser.add_argument("--engine", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.onnx or args.engine:
        print("note: --onnx/--engine are ignored; paths now come from "
              "models/MANIFEST.json (model key %r)" % args.model)

    if not BUILDER.is_file():
        print("missing %s" % BUILDER, file=sys.stderr)
        return 2

    if not args.no_wait:
        wait_for_idle(args.min_avail_mb, args.wait_timeout_s)

    if FETCH.is_file():
        rc = run(["bash", FETCH, args.model], cwd=str(ROOT))
        if rc != 0:
            print("fetch failed for %s" % args.model, file=sys.stderr)
            return rc
    else:
        print("warning: %s missing; assuming the ONNX is already in models/" % FETCH)

    cmd = [sys.executable, BUILDER, "--only", args.model]
    if args.force:
        cmd.append("--force")
    return run(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    sys.exit(main())
