#!/usr/bin/env python3
"""Download YOLOv5n ONNX and build a TensorRT engine when the Jetson is idle.

Official YOLOv8n ONNX URLs 404 as of 2026-09-09; YOLOv5n v7.0 ONNX is the
working default. The ADAS decoder accepts both YOLOv5 (N,85) and YOLOv8 (84,N).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ONNX_URLS = [
    "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx",
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx",
    "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.onnx",
]
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
BUSY_CMDS = ("dms.app", "trtexec", "yolo", "train.py")


def available_mb() -> int:
    meminfo = Path("/proc/meminfo").read_text()
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def gpu_busy() -> bool:
    try:
        out = subprocess.check_output(
            ["timeout", "1", "tegrastats", "--interval", "500"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    line = out.strip().splitlines()[-1] if out.strip() else ""
    if "GR3D_FREQ" not in line:
        return False
    # e.g. GR3D_FREQ 76%
    for tok in line.split():
        if tok.endswith("%") and "GR3D" not in tok:
            continue
    import re

    m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    return bool(m) and int(m.group(1)) >= 20


def other_heavy_process() -> str:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except subprocess.CalledProcessError:
        return ""
    me = str(os.getpid())
    for line in out.splitlines():
        if me in line.split(None, 1)[:1]:
            continue
        for needle in BUSY_CMDS:
            if needle in line and "build_yolo_engine" not in line:
                return line.strip()
    return ""


def wait_for_idle(min_avail_mb: int, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while True:
        avail = available_mb()
        busy = other_heavy_process()
        gpu = gpu_busy()
        if avail >= min_avail_mb and not busy and not gpu:
            print("idle: avail=%sMB gpu_idle=1" % avail)
            return
        if time.time() >= deadline:
            raise SystemExit(
                "timeout waiting for idle (avail=%sMB busy=%r gpu=%s)"
                % (avail, busy, gpu)
            )
        print(
            "waiting: avail=%sMB need=%sMB busy=%s gpu=%s"
            % (avail, min_avail_mb, busy or "-", gpu)
        )
        time.sleep(15)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("downloading %s -> %s" % (url, dest))
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="models/yolov5n.onnx")
    parser.add_argument("--engine", default="models/yolov5n.engine")
    parser.add_argument("--min-avail-mb", type=int, default=2500)
    parser.add_argument("--wait-timeout-s", type=int, default=3600)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    if not args.no_wait:
        wait_for_idle(args.min_avail_mb, args.wait_timeout_s)

    onnx = Path(args.onnx)
    engine = Path(args.engine)
    if not onnx.exists():
        last_err = None
        for url in ONNX_URLS:
            try:
                download(url, onnx)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
        if last_err is not None:
            print("failed to download YOLO onnx: %s" % last_err, file=sys.stderr)
            return 1

    if engine.exists():
        print("engine already exists: %s" % engine)
        return 0

    if not Path(TRTEXEC).exists():
        print("trtexec not found at %s" % TRTEXEC, file=sys.stderr)
        return 1

    log_path = engine.with_suffix(".engine.build.log")
    cmd = [
        TRTEXEC,
        "--onnx=%s" % onnx,
        "--saveEngine=%s" % engine,
        "--fp16",
        "--memPoolSize=workspace:256M",
    ]
    print("running: %s" % " ".join(cmd))
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print("trtexec failed, see %s" % log_path, file=sys.stderr)
        return proc.returncode
    print("wrote %s" % engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
