# ADAS on this Jetson

Last verified on hardware: **2026-09-13, 16:47–17:00 KST**. Workspace
`/home/nvidia/myspace/ADAS`, branch `production-v1`, version 0.3.0 plus an
unreleased hardening pass. Every number below was re-taken in that window with the
GPU mutex held and the board carrying loadavg 2.6–4.0 from other agents' work.

This is the document to read before doing anything on this board.

## Hardware

| Item | Value |
|---|---|
| Board | NVIDIA Jetson Xavier NX Developer Kit |
| L4T / JetPack | R35.6.5 / 5.1.6 |
| RAM | 6.7 GiB total. Typically ~1.1 GiB free, ~4 GiB "available" |
| Power | MODE_20W_6CORE |
| CUDA / TensorRT / cuDNN | 11.4 / 8.5.2.2 / 8.6 |
| Host Python | **3.8.10**. No 3.10. No `python3-venv` (sudo needs a password) |
| OpenCV | 4.5.4 system build — GStreamer **yes**, CUDA **no** |
| ROS 2 | **Not installed.** `rclpy` is absent; only Docker images exist |
| Camera | **None.** `/dev/video*` is absent. Argus plugins are installed |

**Everything is validated by replaying `Ultra-Fast-Lane-Detection-v2/example.mp4`.**
No frame in this project has ever come from a camera.

Neighbouring projects share the board and the GPU:

* `/home/nvidia/myspace/DMS` — TensorRT + GStreamer, ~0.8 GB + GPU when replaying
* `/home/nvidia/myspace/autoJetsonBot` — ROS 2 Humble in Docker
* `/home/nvidia/myspace/cv-research` — FPS/power experiments

**Hold the mutex for anything that touches the GPU:**

```bash
flock /tmp/jetson-gpu.lock -c "<the command>"
```

Do not `pip install` torch, tensorflow, onnx, onnxruntime, pycuda,
opencv-python, ultralytics or jetson-stats. That is a hard project rule: pip
OpenCV shadows the GStreamer build, and models must arrive as prebuilt `.onnx`
and be converted with on-board `trtexec`.

## What works now

* **Detection** — YOLOX-Nano 416 FP16 (Apache-2.0). Default classes are person,
  bicycle, car, motorcycle, bus, train, truck. Decode verified bit-identical to
  YOLOX's own postprocess on frames 150/300/380 of `example.mp4`.
* **Lane** — UFLD-v2 CULane ResNet-18 FP16 with the canonical preprocessing.
  YOLOP is built and works (lane + drivable area + vehicles) but must be
  scheduled at 2–5 Hz.
* **Depth** — MiDaS v2.1 small, **demoted**. It is wired, it is off by default
  (`depth.backend: off`), and even when switched on it publishes **no metric range**:
  every box comes back `RangeSource.UNAVAILABLE`, confidence 0. Its measured
  per-object rank correlation with the reference range on this clip is 0.14–0.25
  against the 0.80 bar it now audits itself against. There is currently **no
  independent range channel on this board** — the arbiter runs on pinhole alone.
* **Tracking** — Kalman range filter + Hungarian association, M-of-N
  confirmation, per-class height priors, truncation flags, metric lateral offset.
* **Planning / control** — constant time-gap ACC with a separate AEB stage, a
  speed-scheduled lateral law, a stateful PI controller with jerk and pedal-rate
  shaping.
* **Safety** — the arbiter is authoritative and independent; its command is what
  the pipeline returns.
* **Operations** — `/healthz`, `/readyz`, `/metrics` on `127.0.0.1:8090`, a JSONL
  event log, systemd notify + watchdog, SIGTERM/SIGHUP handling, `--frames 0`.
* **859 tests pass, 1 fails** on host Python 3.8 (50.75 s). The failure is
  `tests/test_integration.py::test_record_and_replay_round_trip` and it is a real
  defect in `src/adas/tools/replayer.py`, not a flake — see item 11 below.

### Measured, 2026-09-13 16:47–17:00 KST

All unpaced (`--fps 0`), GPU mutex held, board at loadavg 2.6–4.0.

200 frames of `example.mp4`, YOLOX-Nano + UFLD-v2, `--ego-source simulated`:

```
frames=200 failures=0 dropped=0 reconnects=0 settle=1 elapsed=12.49s
measured=16.01 FPS busy=16.16 FPS (61.9 ms/frame)
detections=136  tracks=154  lane detected in 100% of frames
safety: 58 violations, 15 warnings
```

Per-stage (ms, mean / p50 / p95): detect 12.68 / 12.37 / 15.32 ·
lane 41.17 / 38.73 / 46.82 · track 0.95 / 0.11 / 3.04 · plan 0.21 / 0.18 / 0.29 ·
control 0.08 / 0.07 / 0.10 · arbitrate 0.46 / 0.29 / 0.97 ·
frame interval 61.38 / 58.56 / 75.06.

The decision layer costs **1.70 ms/frame mean**. The budget is entirely perception,
and within it, entirely UFLD-v2's host-side pre/post-processing: 16.5 ms of that
41.2 ms is GPU compute (`trtexec`), the rest is CPU.

Backend matrix, same session, same clip:

| detector + lane | frames | FPS | ms/frame |
|---|---:|---:|---:|
| YOLOX-Nano + UFLD-v2 | 200 | 16.01 | 61.9 |
| YOLOX-Nano + UFLD-v2 + MiDaS | 100 | 16.81 | 59.1 |
| YOLOX-Nano + YOLOP (auto `every_n_frames=4`) | 100 | 26.42 | 37.4 |
| YOLOv5n + UFLD-v2 (AGPL, dev only) | 100 | 13.71 | 72.4 |
| YOLOX-Nano + TwinLiteNet | — | refuses to start (no engine) | — |
| mock + mock (no GPU) | 200 | 285.86 | 3.0 |

Memory, measured on a live 300-frame run with the health endpoint up:
`adas_ram_mb 1091.6`, `VmRSS 1.07 GiB`, `VmHWM 1.32 GiB`.

## What does NOT work yet

1. **No camera.** The CSI GStreamer path is written and has never been executed.
2. **The camera calibration is an assumption and is wrong for `example.mp4`.**
   The lane vanishing point over 402 frames puts the horizon at v = 423 px
   (pitch −3.87°), not the 328 px the shipped `pitch_deg: 2.0` implies. With the
   shipped extrinsics the ground-plane lane width goes 2.86 m at 3 m range →
   0.50 m at 10 m, which is impossible. `calibrated: false` caps every derived
   confidence at 0.6 and keeps the metric steering law disengaged, but nothing
   forces a calibration. `geometry.LaneCalibrator` exists and nothing calls it.
3. **No ego speed source.** No CAN interface on this board. `ego.source` defaults
   to `none`, so the planner stays degraded and the arbiter commands a
   minimum-risk manoeuvre — correct, and loud. `--ego-source simulated` closes
   the loop against a point-mass plant (with a 0.15 s actuator lag, without which
   a stepped pedal command produces 40–60 m/s³ of jerk and the arbiter correctly
   sits in LIMITED forever). `--ego-file` replays a recorded channel and is the
   only source that reports `measured = True`; with a realistic 0.1–0.5 s spaced
   CSV, 50 frames of `example.mp4` ran with **0 safety violations**.
4. **UFLD-v2 is 413 MB and needs one contiguous allocation.** It fails to load
   with `Cuda Runtime (out of memory)` under normal memory pressure if a smaller
   engine was loaded first. The pipeline mitigates this by loading the largest
   engine first (measured: works at ~1.1 GB available in that order, fails in the
   other). It does not remove the risk. Check `free -m` first.
5. **No overlay or saved debug video.** Still the highest-value debugging gap.
5b. **No soak test.** The longest run against this codebase is a few hundred frames
   — about 25 seconds of clip. No 8 h result, no RSS curve over time, no FD audit.
6. **ROS 2 bridge never executed.** `rclpy` is not installed. The module imports
   fine and refuses to construct a node.
7. **The systemd unit has never been started**, and
   `deploy/Dockerfile.jetson` has never been built. `sudo` needs a password.
8. **YOLOX-Nano has lower recall than YOLOv5n on the first ~120 frames** of
   `example.mp4` (5 vs 40 road-user detections at conf 0.30), though the
   200-frame totals are comparable. That is a real model difference, not a decode
   bug. `yolox_tiny.engine` is built and is the same code path if recall matters
   more than 2 ms.
9. **TwinLiteNet has no engine.** The decoder is written and unexecuted; the
   backend returns the honest unavailable stub.
10. **Nothing is calibrated against ground truth.** Sign conventions, detector
    thresholds and every planner/arbiter constant are engineering defaults tuned
    on one flat-highway clip.
11. **`tests/test_integration.py::test_record_and_replay_round_trip` fails.**
    `RecordedDetector.infer` reads `getattr(frame, "frame_id")` but
    `ADASPipeline.step` hands it `frame.rgb`, so every replayed frame returns zero
    detections and the replay re-runs the decision layers on an empty road. The
    record/replay tooling is not a usable regression harness until that is fixed.
12. **The `camera_uncalibrated` safety violation is implemented but not wired.**
    `src/adas/runtime/pipeline.py` does not pass `camera.calibrated` into the
    `SafetyContext`, so a live run on the shipped `calibrated: false` camera reports
    `safety=nominal` while acting on assumed metric ranges. `/healthz` does carry the
    note `"camera is not calibrated: metric outputs are assumptions"` — verified in a
    live run today — but the arbiter itself raises nothing.
13. **The arbiter logs one WARNING per frame** in a steady non-nominal state.
    Measured today: exactly 200 `adas.control.arbiter` lines in a 200-frame run
    (270 log lines total). At 20 Hz that is 72,000 lines an hour. The planner layers
    were fixed with a `LogGate`; `arbiter.py` has not been converted.
14. **Several safety thresholds are unreachable from a YAML/JSON config**, including
    two that can latch the terminal DISENGAGE state. See `DEPLOYMENT.md`.

## How to run

```bash
cd ~/myspace/ADAS

# Tests
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m pytest tests/ -q"

# Real engines, blank frames: proves the engines deserialise and the loop runs
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m adas.cli --frames 10"

# Real engines on the clip, closed loop against a simulated plant
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m adas.cli \
  --detector yolox --lane ufld \
  --source Ultra-Fast-Lane-Detection-v2/example.mp4 \
  --ego-source simulated --ego-speed 15 --frames 200 --fps 0"

# YOLOP instead (lane + drivable area). Automatically drops to every_n_frames=4.
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m adas.cli \
  --lane yolop --source Ultra-Fast-Lane-Detection-v2/example.mp4 --frames 100"

# Independent range cross-check on
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m adas.cli \
  --depth midas --source Ultra-Fast-Lane-Detection-v2/example.mp4 --frames 100"

# No GPU / no engines
PYTHONPATH=src python3 -m adas.cli --detector mock --lane mock --allow-mock --frames 20
```

Health while it runs:

```bash
curl -s http://127.0.0.1:8090/healthz | python3 -m json.tool
curl -s http://127.0.0.1:8090/metrics | grep adas_safety_state
```

The CLI flags: `--config --source --detector --lane --depth --ego-source
--ego-speed --ego-file --frames --fps --loop --allow-mock --no-health
--no-events --log-level --log-format --print-config`. Environment:
`ADAS_CONFIG_PATH`, `ADAS_LOG_LEVEL`, `ADAS_LOG_FORMAT`, `ADAS_ALLOW_MOCK`.

## Engines on this board

All FP16, all fully static, all verified to load. Build them with the board idle.

| file | size | GPU median | licence |
|---|---|---|---|
| `models/yolox_nano.engine` | 3.2 MB | 4.69 ms | Apache-2.0 |
| `models/yolox_tiny.engine` | 12.7 MB | 6.43 ms | Apache-2.0 |
| `models/yolov5n.engine` | 5.7 MB | 7.23 ms | **AGPL-3.0-only** |
| `models/ufldv2_culane_res18.engine` | **413.4 MB** | 16.53 ms | MIT code / CULane research-only data |
| `models/yolop_640.engine` | 20.2 MB | 26.98 ms | MIT code / BDD100K research-only data |
| `models/midas_v21_small_256.engine` | 33.9 MB | 6.33 ms | MIT |

```bash
bash scripts/fetch_models.sh                          # sha256-pinned
flock /tmp/jetson-gpu.lock -c "python3 scripts/build_engines.py --only yolox_nano"
```

`models/MANIFEST.json` is the source of truth: URL, sha256, licence, binding
contract and preprocessing formula for every model. `models/README.md` is the
human-readable companion. Never commit `.onnx` or `.engine`.

An engine is version- and device-locked. If TensorRT or the GPU changes, every
engine must be rebuilt, and `TrtEngine` says so in its deserialisation error.

## Courtesy on a shared board

* `flock /tmp/jetson-gpu.lock` around everything that loads TensorRT or runs
  `trtexec`. Other agents work here concurrently.
* Wait for `MemAvailable ≥ 2500 MB` and `GR3D_FREQ ≈ 0%` before a build.
* Pass `--memPoolSize=workspace:512` to `trtexec` to stay inside the ceiling.
* Do not run ADAS TensorRT while `dms.app` is replaying.

## Known local mess

* A root-owned `.pytest_cache/` and some `__pycache__` directories from an old
  Docker run cannot be removed without sudo. `pyproject.toml` redirects pytest's
  cache to `/tmp/adas-pytest-cache`, so they are inert.
* `tests/test.txt` is a stray prose file inside `tests/` claiming "32 passed".
  The real number today is 859 passed / 1 failed.
* Running with the default config writes `data/events.jsonl` into the repository
  (the lab path for the event log). Use `--no-events`, or set `events.path`, if
  that is unwelcome.

## Next, in order

0. **Fix `src/adas/tools/replayer.py`** so the record/replay round trip actually
   replays perception. It is one failing test today, but the consequence is that the
   only offline regression harness this project has is silently comparing every
   change against an empty road.
1. **Calibrate the camera** and wire `LaneCalibrator` into a startup routine or a
   `scripts/calibrate_camera.py`. Everything metric is blocked on this, and the
   measured self-consistency improvement is large (lane width 3.86 ± 0.346 m →
   3.646 ± 0.135 m). While you are there, pass `camera.calibrated` into the
   `SafetyContext` from `pipeline.py` — one line, and it turns on a safety violation
   that is already written and tested.
2. **Overlay and saved debug video** — boxes, track ids, range, lane, plan,
   arbiter state, throttle/brake. Same idea as the DMS replay tool.
3. **A real ego speed channel**, even a recorded one, so the closed loop is not
   closed against a point-mass model.
4. **Start the systemd unit once, under `journalctl -u adas -b -f`,** and build
   `deploy/Dockerfile.jetson` once.
5. **Tune on more than one clip.** Every threshold is currently an engineering
   default validated on flat highway.
6. Only then: live camera, then vehicle/CAN.
