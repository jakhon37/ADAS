#!/usr/bin/env python3
"""Build every ADAS TensorRT engine described by ``models/MANIFEST.json``.

This is the general model builder for the project. It supersedes the
YOLO-only ``scripts/build_yolo_engine.py`` (which now delegates here) and
covers object detection (YOLOv5n, YOLOX-Nano/Tiny), lane detection (UFLDv2),
panoptic driving perception (YOLOP) and monocular relative depth (MiDaS).

Units and conventions
---------------------
* Latency is **GPU compute time in milliseconds**, as reported by ``trtexec``'s
  own timing loop on this board. It excludes H2D/D2H copies and all CPU work.
* Workspace sizes are **MiB**.
* Shapes are NCHW, batch 1, **static**. ``adas.infer.trt_engine.TrtEngine``
  rejects dynamic shapes, so any ONNX with a dynamic axis must declare a
  ``engine_build.static_shapes`` entry in the manifest; min == opt == max.

Shared-board concurrency
------------------------
Several agents and the DMS project share one Jetson Xavier NX. Every TensorRT
invocation -- build, verify and introspect -- is serialised behind ``flock`` on
``/tmp/jetson-gpu.lock`` (override with ``$JETSON_GPU_LOCK``) so two processes
can never contend for the GPU or exceed the ~2.5 GiB free-RAM ceiling. The lock
is held only for the lifetime of the child process.

Failure behaviour
-----------------
Nothing fails silently and nothing is swallowed.

* ONNX missing      -> job recorded under ``blocked`` as ``missing``; other jobs
                       still run; exit status 1.
* ``trtexec`` != 0  -> job recorded under ``blocked`` as ``build_failed`` with
                       the last 25 log lines; exit status 1.
* verify fails      -> job recorded under ``blocked`` as ``verify_failed``;
                       ``verified`` is set to ``false``; exit status 1.
* manifest write    -> atomic (temp file + ``os.replace``); a crash mid-write
                       cannot truncate the manifest.

Usage (Python 3.8 on the board)::

    python3 scripts/build_engines.py --list
    python3 scripts/build_engines.py                 # everything not yet built
    python3 scripts/build_engines.py --only yolop --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
MANIFEST_PATH = os.path.join(MODELS_DIR, "MANIFEST.json")
TRTEXEC = os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec")
GPU_LOCK = os.environ.get("JETSON_GPU_LOCK", "/tmp/jetson-gpu.lock")
FLOCK = "/usr/bin/flock"

#: Free-RAM floor (MiB) below which a build is very likely to be OOM-killed.
#: We warn rather than refuse, because the GPU lock usually frees a neighbour
#: process before our child actually starts allocating.
MIN_AVAIL_MB = 2200


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    sys.stdout.write("[%s] %s\n" % (_ts(), msg))
    sys.stdout.flush()


def sha256_file(path: str) -> str:
    """Lowercase hex SHA-256 of *path*, streamed in 1 MiB chunks.

    Streaming matters: the UFLDv2 ONNX is 825 MB and must not be slurped into
    RAM on a board with ~2.5 GiB free.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def available_mb() -> int:
    """MemAvailable in MiB. Returns 0 if /proc/meminfo cannot be parsed."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError) as exc:
        log("WARNING: cannot read MemAvailable (%s); proceeding without the check" % exc)
        return 0
    return 0


def flock_wrap(cmd: Sequence[str]) -> List[str]:
    """Serialise *cmd* behind the board-wide GPU mutex.

    If ``flock(1)`` is absent the command is returned unchanged and a loud
    warning is printed -- an unserialised TensorRT build on this shared board
    can OOM a neighbouring project, so the operator is told rather than left to
    discover it from a corrupted run.
    """
    if not os.path.isfile(FLOCK):
        log("WARNING: %s not found - running WITHOUT the shared GPU mutex" % FLOCK)
        return list(cmd)
    # util-linux flock is `flock <file> <command> [args...]`. It does NOT accept
    # a `--` separator before the command: it would try to exec "--".
    return [FLOCK, GPU_LOCK] + list(cmd)


# ---------------------------------------------------------------------------
# trtexec log parsing
# ---------------------------------------------------------------------------

# trtexec 8.5 prints, after deserialising the engine it just built:
#   [I] Created input binding for images with dimensions 1x3x640x640
_BINDING_RE = re.compile(
    r"Created\s+(input|output)\s+binding\s+for\s+(\S+)\s+with\s+dimensions\s+([0-9x]+)"
)
_ENGINE_BUILT_RE = re.compile(r"Engine built in\s+([0-9.]+)\s+sec")
_LATENCY_KEYS = (
    ("gpu_compute_min_ms", r"GPU Compute Time:.*?\bmin\s*=\s*([0-9.]+)\s*ms"),
    ("gpu_compute_max_ms", r"GPU Compute Time:.*?\bmax\s*=\s*([0-9.]+)\s*ms"),
    ("gpu_compute_mean_ms", r"GPU Compute Time:.*?\bmean\s*=\s*([0-9.]+)\s*ms"),
    ("gpu_compute_median_ms", r"GPU Compute Time:.*?\bmedian\s*=\s*([0-9.]+)\s*ms"),
    ("end_to_end_median_ms", r"^\s*\[[^\]]*\]\s+\[I\]\s+Latency:.*?\bmedian\s*=\s*([0-9.]+)\s*ms"),
    ("throughput_qps", r"Throughput:\s*([0-9.]+)\s*qps"),
)
_LATENCY_RES = [(k, re.compile(p, re.MULTILINE)) for k, p in _LATENCY_KEYS]


def parse_bindings(log_text: str) -> Dict[str, List[Dict[str, object]]]:
    """Binding names and shapes from a trtexec log.

    Returns ``{"inputs": [...], "outputs": [...]}`` preserving trtexec's own
    binding order, which is the order ``TrtEngine`` will enumerate. Duplicate
    names keep their first position but take the latest shape, because trtexec
    may print the set twice (build then load).
    """
    inputs: "Dict[str, Dict[str, object]]" = {}
    outputs: "Dict[str, Dict[str, object]]" = {}
    for line in log_text.splitlines():
        m = _BINDING_RE.search(line)
        if not m:
            continue
        kind, name, dims_txt = m.groups()
        dims = [int(t) for t in dims_txt.split("x") if t.isdigit()]
        bucket = inputs if kind == "input" else outputs
        rec = bucket.setdefault(name, {"name": name})
        rec["shape"] = dims
    return {"inputs": list(inputs.values()), "outputs": list(outputs.values())}


def parse_latency(log_text: str) -> Dict[str, float]:
    """GPU compute statistics (ms) and throughput (qps) from a trtexec log."""
    out: Dict[str, float] = {}
    for key, rx in _LATENCY_RES:
        m = rx.search(log_text)
        if m:
            out[key] = float(m.group(1))
    m = _ENGINE_BUILT_RE.search(log_text)
    if m:
        out["engine_build_seconds"] = float(m.group(1))
    return out


# ---------------------------------------------------------------------------
# engine introspection (authoritative dtypes)
# ---------------------------------------------------------------------------

_INTROSPECT_SRC = r"""
import json, sys
import tensorrt as trt
path = sys.argv[1]
logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)
with open(path, "rb") as fh:
    engine = runtime.deserialize_cuda_engine(fh.read())
if engine is None:
    raise SystemExit("deserialize failed")
inputs, outputs = [], []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    rec = {
        "name": name,
        "shape": [int(d) for d in engine.get_tensor_shape(name)],
        "dtype": str(engine.get_tensor_dtype(name)).split(".")[-1],
        "index": i,
    }
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        inputs.append(rec)
    else:
        outputs.append(rec)
print(json.dumps({"inputs": inputs, "outputs": outputs, "trt": trt.__version__}))
"""


def introspect_engine(engine_path: str) -> Optional[Dict[str, object]]:
    """Enumerate the built engine's IO tensors with their real dtypes.

    Runs the TensorRT Python runtime in a child process behind the GPU mutex so
    a deserialisation crash cannot take down the builder. Returns ``None`` and
    logs the reason when introspection is unavailable (e.g. no ``tensorrt``
    module); the caller then falls back to the shapes parsed from the log.
    """
    cmd = flock_wrap([sys.executable, "-c", _INTROSPECT_SRC, engine_path])
    try:
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log("WARNING: engine introspection failed to launch for %s: %s"
            % (os.path.basename(engine_path), exc))
        return None
    if out.returncode != 0:
        log("WARNING: engine introspection rc=%d for %s: %s"
            % (out.returncode, os.path.basename(engine_path),
               out.stderr.decode("utf-8", "replace").strip()[-500:]))
        return None
    try:
        return json.loads(out.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        log("WARNING: unparsable introspection output for %s: %s"
            % (os.path.basename(engine_path), exc))
        return None


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def load_manifest() -> Dict[str, object]:
    if not os.path.isfile(MANIFEST_PATH):
        raise SystemExit(
            "%s not found. It is hand-authored and version-controlled; restore it "
            "before building." % os.path.relpath(MANIFEST_PATH, ROOT)
        )
    with open(MANIFEST_PATH) as fh:
        data = json.load(fh)
    data.setdefault("models", {})
    data.setdefault("blocked", {})
    return data


def save_manifest(data: Dict[str, object]) -> None:
    """Atomically rewrite MANIFEST.json (temp file + rename)."""
    data["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, MANIFEST_PATH)


README_PATH = os.path.join(MODELS_DIR, "README.md")
README_START = "<!-- LATENCY TABLE START -->"
README_END = "<!-- LATENCY TABLE END -->"


def _fmt_ms(value: object) -> str:
    return "%.2f" % value if isinstance(value, (int, float)) else "-"


def _fmt_mb(value: object) -> str:
    return "%.1f" % (value / 1e6) if isinstance(value, (int, float)) else "-"


def update_readme_latency(manifest: Dict[str, object]) -> None:
    """Regenerate the measured-performance table in ``models/README.md``.

    Rewrites only the region between the ``LATENCY TABLE`` HTML comments so the
    prose around it is never touched. Silently does nothing when the README or
    its markers are absent -- the table is documentation, and a missing marker
    must not fail a build -- but says so on stdout either way.
    """
    if not os.path.isfile(README_PATH):
        log("note: %s absent; skipping the latency table" % os.path.relpath(README_PATH, ROOT))
        return
    with open(README_PATH) as fh:
        text = fh.read()
    if README_START not in text or README_END not in text:
        log("note: latency-table markers missing in models/README.md; not editing it")
        return

    rows = [
        "| Model | Engine | Size (MB) | GPU compute median (ms) | GPU compute mean (ms) | "
        "Throughput (qps) | Build (s) | Verified |",
        "|---|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for name, entry in manifest["models"].items():
        lat = entry.get("verify") or {}
        if not isinstance(lat, dict) or "gpu_compute_median_ms" not in lat:
            lat = entry.get("build_latency") or {}
        rows.append("| `%s` | `%s` | %s | %s | %s | %s | %s | %s |" % (
            name,
            entry.get("engine_file", "-"),
            _fmt_mb(entry.get("engine_bytes")),
            _fmt_ms(lat.get("gpu_compute_median_ms")),
            _fmt_ms(lat.get("gpu_compute_mean_ms")),
            _fmt_ms(lat.get("throughput_qps")),
            _fmt_ms((entry.get("build_latency") or {}).get("engine_build_seconds")),
            "yes" if entry.get("verified") else "no",
        ))
    rows.append("")
    rows.append("Measured on %s, TensorRT %s, FP16, batch 1. Regenerate with "
                "`python3 scripts/build_engines.py`." % (
                    manifest.get("device", "this board"),
                    manifest.get("trt_version", "unknown")))

    head, _, rest = text.partition(README_START)
    _, _, tail = rest.partition(README_END)
    new_text = head + README_START + "\n" + "\n".join(rows) + "\n" + README_END + tail
    tmp = README_PATH + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(new_text)
    os.replace(tmp, README_PATH)
    log("updated the latency table in models/README.md")


def trt_version() -> str:
    """Full TensorRT version on this board, e.g. ``8.5.2.2``.

    Prefers ``tensorrt.__version__`` (the only source that carries the build
    component); falls back to decoding trtexec's ``[TensorRT vMMMNN]`` banner,
    which encodes major * 1000 + minor * 100 + patch and therefore loses the
    build number. Returns ``"unknown"`` if neither is readable -- the manifest
    then says so rather than claiming a version it did not observe.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import tensorrt; print(tensorrt.__version__)"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120,
        )
        if out.returncode == 0:
            ver = out.stdout.decode("utf-8", "replace").strip()
            if ver:
                return ver
    except (subprocess.SubprocessError, OSError) as exc:
        log("note: tensorrt module not importable for a version check (%s)" % exc)

    try:
        banner = subprocess.run(
            [TRTEXEC, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=120,
        ).stdout.decode("utf-8", "replace")
    except (subprocess.SubprocessError, OSError) as exc:
        log("WARNING: could not read the trtexec version: %s" % exc)
        return "unknown"
    m = re.search(r"\[TensorRT v(\d+)\]", banner)
    if not m:
        return "unknown"
    raw = m.group(1)  # "8502" -> major 8, minor 5, patch 02
    if len(raw) >= 4:
        return "%d.%d.%d" % (int(raw[:-3]), int(raw[-3]), int(raw[-2:]))
    return raw


# ---------------------------------------------------------------------------
# build / verify
# ---------------------------------------------------------------------------


def build_cmd(entry: Dict[str, object], onnx: str, engine: str) -> List[str]:
    build_cfg = entry.get("engine_build") or {}
    workspace = int(build_cfg.get("workspace_mib", 512))
    cmd = [
        TRTEXEC,
        "--onnx=%s" % onnx,
        "--saveEngine=%s" % engine,
        "--memPoolSize=workspace:%d" % workspace,
        "--verbose",
    ]
    if str(entry.get("precision", "fp16")).lower() == "fp16":
        cmd.append("--fp16")
    for tensor, dims in (build_cfg.get("static_shapes") or {}).items():
        spec = "x".join(str(int(d)) for d in dims)
        cmd += ["--minShapes=%s:%s" % (tensor, spec),
                "--optShapes=%s:%s" % (tensor, spec),
                "--maxShapes=%s:%s" % (tensor, spec)]
    for extra in (build_cfg.get("extra_args") or []):
        cmd.append(str(extra))
    return cmd


def run_build(name: str, entry: Dict[str, object], force: bool) -> Dict[str, object]:
    """Build one engine. Returns a status fragment; never raises on trtexec failure."""
    onnx = os.path.join(MODELS_DIR, str(entry["onnx_file"]))
    engine = os.path.join(MODELS_DIR, str(entry["engine_file"]))
    log_path = engine + ".build.log"

    if not os.path.isfile(onnx):
        return {
            "status": "missing",
            "error": "ONNX not present: %s. Run: bash scripts/fetch_models.sh"
                     % os.path.relpath(onnx, ROOT),
        }

    if os.path.isfile(engine) and not force:
        log("%s: engine already present, not rebuilding (--force to override)" % name)
        frag: Dict[str, object] = {"status": "ok", "reused": True}
        if os.path.isfile(log_path):
            with open(log_path, errors="replace") as fh:
                text = fh.read()
            frag["bindings"] = parse_bindings(text)
            frag["latency"] = parse_latency(text)
        return frag

    avail = available_mb()
    if avail and avail < MIN_AVAIL_MB:
        log("%s: only %d MiB available (want >= %d MiB). Queuing behind the GPU "
            "mutex anyway; if this OOMs, retry when the board is idle."
            % (name, avail, MIN_AVAIL_MB))

    cmd = build_cmd(entry, onnx, engine)
    log("%s: building %s (%.1f MB ONNX)" % (name, os.path.basename(engine),
                                            os.path.getsize(onnx) / 1e6))
    log("%s: %s" % (name, " ".join(cmd)))
    t0 = time.monotonic()
    with open(log_path, "w") as log_fh:
        proc = subprocess.run(flock_wrap(cmd), stdout=log_fh, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - t0

    with open(log_path, errors="replace") as fh:
        text = fh.read()

    if proc.returncode != 0 or not os.path.isfile(engine):
        tail = "".join(text.splitlines(True)[-25:]).strip()
        sys.stderr.write("%s: trtexec failed rc=%s\n%s\n" % (name, proc.returncode, tail))
        return {
            "status": "build_failed",
            "returncode": proc.returncode,
            "wall_seconds": round(elapsed, 1),
            "log": os.path.relpath(log_path, ROOT),
            "error": tail[-4000:],
        }

    latency = parse_latency(text)
    log("%s: built in %.0f s wall, engine %.1f MB, GPU compute median %s ms"
        % (name, elapsed, os.path.getsize(engine) / 1e6,
           latency.get("gpu_compute_median_ms", "?")))
    return {
        "status": "ok",
        "wall_seconds": round(elapsed, 1),
        "log": os.path.relpath(log_path, ROOT),
        "bindings": parse_bindings(text),
        "latency": latency,
    }


def verify_engine(name: str, entry: Dict[str, object]) -> Dict[str, object]:
    """Round-trip the saved plan through ``trtexec --loadEngine``.

    This is the only check that catches a truncated or version-mismatched plan
    before the runtime does. Returns ``{"loaded": bool, ...}`` plus the latency
    measured on the reloaded engine; on failure it also carries the log tail.
    """
    engine = os.path.join(MODELS_DIR, str(entry["engine_file"]))
    if not os.path.isfile(engine):
        return {"loaded": False, "error": "engine file missing"}
    workspace = int((entry.get("engine_build") or {}).get("workspace_mib", 512))
    verify_log = engine + ".verify.log"
    cmd = [
        TRTEXEC,
        "--loadEngine=%s" % engine,
        "--iterations=50",
        "--warmUp=500",
        "--avgRuns=25",
        "--memPoolSize=workspace:%d" % workspace,
    ]
    with open(verify_log, "w") as fh:
        proc = subprocess.run(flock_wrap(cmd), stdout=fh, stderr=subprocess.STDOUT)
    with open(verify_log, errors="replace") as fh:
        text = fh.read()
    ok = proc.returncode == 0 and "&&&& PASSED" in text
    result: Dict[str, object] = {
        "loaded": bool(ok),
        "log": os.path.relpath(verify_log, ROOT),
    }
    result.update(parse_latency(text))
    if not ok:
        result["error"] = "".join(text.splitlines(True)[-20:]).strip()[-2000:]
    return result


def merge_bindings(
    from_log: Dict[str, List[Dict[str, object]]],
    from_engine: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Prefer the engine's own introspection; fall back to the parsed log."""
    if from_engine and from_engine.get("inputs"):
        return {"inputs": from_engine["inputs"], "outputs": from_engine["outputs"],
                "source": "tensorrt_runtime"}
    result = {
        "inputs": from_log.get("inputs", []),
        "outputs": from_log.get("outputs", []),
        "source": "trtexec_log",
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--only", action="append", default=[], metavar="NAME",
                   help="build only this manifest model (repeatable)")
    p.add_argument("--force", action="store_true",
                   help="rebuild even when the .engine already exists")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the trtexec --loadEngine round-trip")
    p.add_argument("--list", action="store_true", help="list manifest models and exit")
    p.add_argument("--readme-only", action="store_true",
                   help="regenerate the models/README.md latency table and exit")
    args = p.parse_args(argv)

    manifest = load_manifest()
    models: Dict[str, Dict[str, object]] = manifest["models"]

    if args.list:
        for name, entry in models.items():
            engine = os.path.join(MODELS_DIR, str(entry["engine_file"]))
            print("%-22s %-8s %-38s -> %-34s %s" % (
                name, entry.get("precision", "?"), entry["onnx_file"],
                entry["engine_file"], "built" if os.path.isfile(engine) else "-"))
        return 0

    if args.readme_only:
        update_readme_latency(manifest)
        return 0

    if not os.path.isfile(TRTEXEC):
        sys.stderr.write("trtexec not found at %s (set $TRTEXEC)\n" % TRTEXEC)
        return 2

    unknown = [n for n in args.only if n not in models]
    if unknown:
        sys.stderr.write("unknown model(s): %s\nknown: %s\n"
                         % (", ".join(unknown), ", ".join(models)))
        return 2
    names = list(args.only) if args.only else list(models)

    manifest["trt_version"] = trt_version()
    blocked: Dict[str, object] = manifest["blocked"]
    failures: List[str] = []

    for name in names:
        entry = models[name]
        frag = run_build(name, entry, force=args.force)

        if frag.get("status") != "ok":
            blocked[name] = {
                "reason": frag.get("status"),
                "error": frag.get("error"),
                "log": frag.get("log"),
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            entry["engine_sha256"] = None
            entry["engine_bytes"] = None
            entry["verified"] = False
            failures.append(name)
            save_manifest(manifest)
            continue

        onnx = os.path.join(MODELS_DIR, str(entry["onnx_file"]))
        engine = os.path.join(MODELS_DIR, str(entry["engine_file"]))
        pinned = entry.get("onnx_sha256")
        actual = sha256_file(onnx)
        if pinned and pinned != actual:
            # A changed ONNX under a pinned hash means the supply chain moved.
            # Record it and fail the job rather than shipping an unknown model.
            blocked[name] = {
                "reason": "onnx_sha256_mismatch",
                "error": "pinned %s but found %s for %s" % (pinned, actual, entry["onnx_file"]),
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            entry["verified"] = False
            failures.append(name)
            save_manifest(manifest)
            continue
        entry["onnx_sha256"] = actual
        entry["onnx_bytes"] = os.path.getsize(onnx)
        entry["engine_sha256"] = sha256_file(engine)
        entry["engine_bytes"] = os.path.getsize(engine)
        if frag.get("log"):
            # A reused engine carries no fresh log; keep the recorded one.
            entry["build_log"] = frag["log"]
        if frag.get("latency"):
            entry["build_latency"] = frag["latency"]
        if frag.get("wall_seconds") is not None:
            entry["build_wall_seconds"] = frag["wall_seconds"]

        introspected = introspect_engine(engine)
        entry["bindings"] = merge_bindings(frag.get("bindings") or {}, introspected)

        if args.no_verify:
            entry["verified"] = False
            entry["verify"] = {"loaded": None, "skipped": True}
        else:
            v = verify_engine(name, entry)
            entry["verified"] = bool(v.get("loaded"))
            entry["verify"] = v
            if v.get("loaded"):
                blocked.pop(name, None)
                log("%s: verified, reload GPU compute median %s ms"
                    % (name, v.get("gpu_compute_median_ms", "?")))
            else:
                blocked[name] = {
                    "reason": "verify_failed",
                    "error": v.get("error"),
                    "log": v.get("log"),
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                failures.append(name)

        save_manifest(manifest)

    save_manifest(manifest)
    update_readme_latency(manifest)
    if failures:
        sys.stderr.write("failed: %s (details under models/MANIFEST.json -> blocked)\n"
                         % ", ".join(sorted(set(failures))))
        return 1
    log("all requested engines built and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
