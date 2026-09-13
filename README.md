# ADAS Core

[![CI](https://github.com/jakhon37/ADAS/workflows/CI/badge.svg)](https://github.com/jakhon37/ADAS/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A camera-only ADAS reference pipeline for the NVIDIA Jetson Xavier NX: TensorRT
perception, a Kalman/Hungarian tracker in ground-plane metres, a constant
time-gap longitudinal law with AEB, a lane-keeping law, and an **authoritative
safety arbiter** whose command is the only one that reaches an actuator.

> ## ⚠️ This is NOT production-ready and has never driven a vehicle
>
> It runs real models on real recorded video on real hardware, and it is honest
> about what it does not know. It is a bring-up and research platform. See
> [What is not production ready](#what-is-not-production-ready) — that list is
> the most important section of this file.

**Version 0.3.0.** Measured on this Xavier NX on 2026-09-13: 200 frames of
`example.mp4` through YOLOX-Nano + UFLD-v2 + tracker + planner + controller +
arbiter at **15.3 FPS end to end**, 0 failures, 743/743 tests passing.

---

## What actually works

| Capability | Backend | State |
|---|---|---|
| Object detection | YOLOX-Nano 416 (TensorRT FP16) | **Real.** Decode verified bit-identical to YOLOX's own postprocess. Default classes: person, bicycle, car, motorcycle, bus, train, truck |
| Object detection (alt) | YOLOX-Tiny 416, YOLOv5n 640 | **Real.** YOLOv5n is AGPL — see [Licensing](#licensing) |
| Lane perception | UFLD-v2 CULane ResNet-18 (TensorRT FP16) | **Real.** Canonical preprocessing, per-dataset anchors, ego boundaries by index |
| Lane + free space | YOLOP (TensorRT FP16) | **Real.** Lane fit is noisier than UFLD; the drivable-area mask is good. Must be scheduled at 2–5 Hz |
| Lane (TwinLiteNet) | — | **Honest stub.** No engine exists on this board; the decoder is written, the constructor refuses to run without weights, and the stub reports `is_mock=True` and never invents a lane |
| Independent range | MiDaS v2.1 small (TensorRT FP16) | **Real, off by default.** Inverse *relative* depth, affine-aligned per frame against road-plane anchors. A cross-check, not a metric sensor |
| Camera geometry | Pinhole + flat-ground homography | **Real, UNCALIBRATED by default.** Every derived metric value is confidence-capped and labelled |
| Tracking | Kalman range filter + Hungarian association | **Real.** M-of-N confirmation, class-keyed height priors, truncation flags, per-track TTC |
| Longitudinal | Constant time-gap ACC + separate AEB stage | **Real** |
| Lateral | Speed-scheduled pixel law; Stanley law behind a calibrated camera | **Real (non-metric path active)**. The metric law refuses to engage on an uncalibrated camera |
| Safety arbitration | Independent arbiter with its own lead selection, range filter and kinematics | **Real, and authoritative** |
| Operations | Prometheus `/metrics`, `/healthz`, `/readyz`, JSONL event log, systemd notify + watchdog | **Real and wired** |
| ROS 2 bridge | `adas.ros2` | **Written, NEVER EXECUTED.** `rclpy` is not installed on this board |
| Mock detector / mock lane | — | **Honest stubs.** `is_mock=True`, loud banners, refused unless `--allow-mock` |

### Measured numbers

All on the Xavier NX (MODE_20W_6CORE), 2026-09-13, GPU mutex held.

**End to end, 200 frames of `Ultra-Fast-Lane-Detection-v2/example.mp4`, YOLOX-Nano
+ UFLD-v2, unpaced:**

```
frames=200 failures=0 dropped=0 reconnects=0 elapsed=13.09s
measured=15.28 FPS  busy=15.44 FPS  (64.8 ms/frame)
detections=136  tracks=154  lane detected in 100% of frames
```

Per-stage latency from that run (ms):

| stage | mean | p50 | p95 | max |
|---|---|---|---|---|
| detect (YOLOX-Nano 416) | 14.05 | 12.73 | 15.64 | 133.5 |
| lane (UFLD-v2) | 42.92 | 40.57 | 54.05 | 131.7 |
| track | 0.93 | 0.11 | 2.83 | 4.29 |
| plan | 0.20 | 0.18 | 0.27 | 1.68 |
| control | 0.07 | 0.06 | 0.12 | 0.27 |
| arbitrate | 0.44 | 0.30 | 0.85 | 1.08 |

The first-frame `max` values are engine warm-up. The decision layer (track +
plan + control + arbitrate) costs **1.6 ms/frame**; the budget is entirely
perception.

**Mock backends (no GPU):** 3.5 ms/frame, ~282 FPS busy.

**Raw engine compute** (`trtexec`, batch 1, FP16 — from `models/README.md`):
YOLOX-Nano 4.69 ms · MiDaS 6.33 ms · YOLOX-Tiny 6.43 ms · YOLOv5n 7.23 ms ·
UFLD-v2 16.53 ms · YOLOP 26.98 ms. The gap between 16.5 ms of UFLD GPU compute
and 42.9 ms of measured lane stage is host-side pre/post-processing plus
`TrtEngine`'s copies.

### Memory, and one load-order rule that matters

`ufldv2_culane_res18.engine` is **413 MB** — UFLD-v2's head is a single ~186 M
parameter dense layer. On the Xavier's unified memory a single contiguous
`cudaMalloc` is what fails, not the total, so **the pipeline loads the largest
engine first** (`adas.cli.build_pipeline` loads lane before detector). Measured
on this board with ~1.1 GB available: lane-then-detector loads; detector-then-lane
fails with `Cuda Runtime (out of memory)`. The factory logs a warning when the
available memory is under 1.6× the engine size, so the failure is diagnosable
instead of a bare TensorRT stack trace.

---

## Quick start on this Jetson

`python3-venv` is not installed; use the system interpreter and `PYTHONPATH`.
Wrap anything that touches the GPU in the board's mutex.

```bash
cd ~/myspace/ADAS

# Tests (no GPU needed for most of them)
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m pytest tests/ -q"

# Real models, blank synthetic frames: proves the engines load and the loop runs
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m adas.cli --frames 10"

# Real models on the replay clip, closed loop against a simulated plant
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m adas.cli \
  --detector yolox --lane ufld \
  --source Ultra-Fast-Lane-Detection-v2/example.mp4 \
  --ego-source simulated --ego-speed 15 \
  --frames 200 --fps 0"

# No engines / no GPU: mock backends, which you must opt into
PYTHONPATH=src python3 -m adas.cli --detector mock --lane mock --allow-mock --frames 20
```

See [QUICKSTART.md](QUICKSTART.md) for more, and [docs/JETSON.md](docs/JETSON.md)
for board-specific notes.

---

## Ego speed: there is no vehicle bus

This board has no CAN interface, and a constant-time-gap law without ego speed is
undefined. Rather than let a fabricated speed leak into the planner, the source is
a configuration choice and every option states what it is:

| `ego.source` | what it is | `EgoState.valid` | measured |
|---|---|---|---|
| `none` *(default)* | no ego speed at all | `False` | no |
| `config` | a constant the operator declared | `True` | no |
| `simulated` | a point-mass plant driven by the **actuated** command | `True` | no |
| `file` | a recorded `timestamp_s,speed_mps` channel (CSV or JSON) | `True` while fresh | **yes** |

With `none`, the planner holds its target and ramps it down and the arbiter
commands a minimum-risk manoeuvre; after `safety.disengage_after_frames` (40) the
arbiter latches **DISENGAGE**, because a missing ego state is a persistent fault,
not traffic. A bare `--frames 200` on the replay clip therefore ends latched in
`disengage` with `violations=ego_state_invalid` — measured, and correct: the
vehicle will not drive, because nothing knows how fast it is going. Pass
`--ego-source simulated` or `--ego-file` to close the loop. `/healthz` publishes
`ego_speed_valid`, which is only true for a source that is both valid *and*
measured, so a fleet query can separate bench units from vehicles.

Between two `file` samples whose gap is at most `ego.max_age_s` the speed is
linearly interpolated (a channel recorded at 10–50 Hz supports that, and holding
the earlier sample instead injects a staircase the arbiter reads as jerk). Across
a wider gap, and past the last sample, the value is held for at most
`ego.max_age_s` and then reported invalid — a frozen speed is exactly what makes
a dead bus look healthy.

With a realistic recorded channel the whole system runs clean: 50 frames of
`example.mp4` with YOLOX-Nano and a 0.1–0.5 s spaced speed CSV gave
**0 safety violations, 0 warnings, `safety=nominal` throughout**, with the
throttle rising from 0.00 to 0.24 as the controller closed on the 15 m/s cruise
target from a measured ~12 m/s.

---

## Safety architecture

```
capture → detect → lane → track → depth cross-check → plan → control → ARBITRATE → actuators
                                                                          ▲
                                        the arbiter's command is the only one that leaves
```

The arbiter (`adas.control.arbiter.SafetyArbiter`) is deliberately independent of
the planner: its own lead selection over the raw track list (ordered by TTC, then
range), its own alpha-beta range filter with a jump gate, its own kinematics from
measured ego speed, its own RSS minimum-gap test. It never reads
`TrackedObject.velocity_mps`. On any violation the command the pipeline returns
is the arbiter's rate-shaped fail-safe, not the controller's request.

Two properties are enforced by tests in `tests/test_pipeline.py`:

* **A runaway controller cannot actuate.**
  `test_arbiter_command_is_what_the_pipeline_returns` installs a controller that
  always commands full throttle and asserts the returned throttle is 0.
* **A perception failure is a fault, not an empty road.** An exception from the
  detector or lane estimator produces `PerceptionStatus(ok=False)` with a rising
  `consecutive_failures`; the tracker is **not** advanced with a synthetic empty
  detection list (which would read as "every object vanished"), and the planner
  is told `perception_valid=False`. An empty detection list from a *working*
  detector still means the road is clear.

When `pipeline.step` raises, the runner actuates
`ADASPipeline.failsafe_command()` — the arbiter's minimum-risk command — and
counts the failure; ten consecutive failures stop the loop. It never latches the
previous command on the actuators.

---

## Configuration

`config.example.json` is the **bench profile**: mock backends, synthetic source,
`allow_mock: true`. It runs on a machine with no engines and doubles as the
complete key reference. Every key is validated, and an unknown key is a hard
error naming the section (a typo in a safety limit used to be silently ignored).

Cross-section rules are checked at load, not discovered at runtime: the planner
may not cruise faster than the safety ceiling, may not steer further than the
arbiter allows, may not follow closer than the absolute minimum gap, and
`safety.max_road_wheel_rad` must equal `radians(controller.max_steering_angle_deg)`
— leave it at `0.0` and it is derived.

### The vehicle profile

Start from `config.example.json` and change these:

```json
{
  "allow_mock": false,
  "detector": { "backend": "yolox", "model_path": "models/yolox_nano.engine" },
  "lane":     { "backend": "ufld",  "model_path": "models/ufldv2_culane_res18.engine" },
  "depth":    { "backend": "midas", "cadence_frames": 5 },
  "camera": {
    "enabled": true, "calibrated": true, "label": "vehicle-2026-09-13",
    "image_width": 1280, "image_height": 720,
    "fx": 910.0, "fy": 910.0, "cx": 640.0, "cy": 360.0,
    "mount_height_m": 1.30, "pitch_deg": -3.87
  },
  "ego":    { "source": "file", "file": "recordings/speed.csv", "max_age_s": 0.15 },
  "source": { "type": "camera", "uri": "csi:0", "width": 1280, "height": 720 },
  "health": { "enabled": true, "port": 8090 },
  "events": { "enabled": true, "fsync": "critical" }
}
```

Do **not** set `"calibrated": true` until you have actually measured the mount
height and pitch. It is the switch that tells the whole stack its metres are
real.

### Calibration

The shipped camera block is an assumption (`fx = fy = 910`, `h = 1.30 m`,
`pitch = 2°`), and it is measurably wrong for `example.mp4`: the lane vanishing
point over 402 frames puts the road-plane horizon at v = 423 px, i.e. a pitch of
−3.87°, not the 328 px the assumed pitch implies. With the assumed extrinsics the
ground-plane lane width goes from 2.86 m at 3 m range to 0.50 m at 10 m, which is
physically impossible. The code is honest about this — `calibrated=False` caps
every derived range confidence at 0.6, keeps `LaneModel.lines[].coeffs` metric
only when a camera exists, and holds the metric steering law disengaged — but
**nothing forces a calibration**, and an integrator who ships the default camera
gets metric numbers that are self-consistently wrong.

`adas.perception.geometry.LaneCalibrator` accumulates lane observations and emits
a calibrated `CameraConfig`. On `example.mp4` it moved the measured lane width
from 3.86 ± 0.346 m (range-inconsistent) to 3.646 ± 0.135 m (range-consistent).
Nothing in the runtime calls it yet; wiring it, or shipping a
`scripts/calibrate_camera.py`, is the single highest-value follow-up.

---

## Operations

The process is supervisable. `adas.io` provides a stdlib-only Prometheus
registry, an HTTP endpoint and a durable event log, and `adas.cli` wires all of
it.

```bash
curl -s http://127.0.0.1:8090/healthz | python3 -m json.tool
curl -s http://127.0.0.1:8090/metrics | grep adas_safety_state
```

* `/livez` — always 200 while the HTTP thread is alive.
* `/readyz` — 200 only when the source is delivering, no engine is missing or
  failed, perception is ok, at least one frame has completed, and the snapshot is
  fresh.
* `/healthz` — 503 when not ok or **stale**; 200 when merely degraded, because
  `LIMITED` is a mode, not an outage. Binds to loopback; a non-loopback bind
  needs `allow_remote` set on purpose, since the body carries live safety state,
  ego speed and lead range.
* `/metrics` — ~55 series including `adas_safety_state{state}`,
  `adas_stage_duration_ms{stage}`, `adas_lane_is_mock`, `adas_ego_speed_valid`
  and `adas_build_info`.

The event log (`data/events.jsonl` in a lab profile) records lifecycle,
safety-state transitions, perception dropouts, engine failures and source events
as JSON lines. Entering `MIN_RISK_MANEUVER` or `DISENGAGE` is CRITICAL and is
`fsync`ed immediately.

`SIGTERM`/`SIGINT` stop the loop after the current frame and unwind through the
normal shutdown path; `SIGHUP` reopens the event log for `logrotate`.
`--frames 0` runs until stopped. `READY=1` is sent to systemd only after the
first frame has completed — readiness means a frame went all the way through, not
that the process started.

Deployment assets (systemd unit, logrotate policy, installer, Jetson container
image) are in [`deploy/`](deploy/README.md). **The service has never been
started**: `sudo` is unavailable in this environment, so the unit is validated
only by `systemd-analyze verify` and review.

Environment variables that are actually read: `ADAS_CONFIG_PATH`,
`ADAS_LOG_LEVEL`, `ADAS_LOG_FORMAT`, `ADAS_ALLOW_MOCK`.

---

## Licensing

The **source** in this repository is MIT. **Model weights are not distributed
with it** — `models/*.onnx` and `models/*.engine` are gitignored and excluded
from the sdist by `MANIFEST.in`. Each model's URL, sha256, licence and
preprocessing contract is recorded in `models/MANIFEST.json`.

| Model | Code licence | Dataset caveat | Shipped as default |
|---|---|---|---|
| YOLOX-Nano / Tiny | Apache-2.0 | COCO | **yes** (detector) |
| UFLD-v2 CULane ResNet-18 | MIT | CULane: research / non-commercial | **yes** (lane) |
| YOLOP | MIT | BDD100K: research / non-commercial | no |
| MiDaS v2.1 small | MIT | — | no (`depth.backend: off`) |
| YOLOv5n | **AGPL-3.0-only** | COCO | **no** |

**The AGPL-vs-MIT conflict and how it is resolved.** `pyproject.toml` declares
MIT while `models/yolov5n.onnx` is AGPL-3.0-only. The resolution is: YOLOv5n is
no longer the default detector (it was), it is not distributed (weights are
gitignored and `MANIFEST.in` excludes every `*.onnx`/`*.engine`), and the
licence-clean YOLOX-Nano replacement is both verified equivalent in decode and
faster — 14.0 ms vs 24.4 ms end to end. YOLOv5n remains buildable as a
development baseline, and a distributed build must not include it. The declared
`license = { text = "MIT" }` therefore covers everything actually shipped.

The dataset caveats are real and the MIT *code* licence does not launder them:
CULane and BDD100K are research/non-commercial datasets, so a commercial
deployment needs weights trained on data you are licensed to use.

---

## What is not production ready

This is the list to read before quoting anything above.

1. **It has never run in a vehicle, and there is no camera on this board.**
   `/dev/video*` is absent. Everything is validated by replaying local mp4 clips.
   The CSI GStreamer path is written and unexecuted.
2. **The camera calibration is an assumption**, and demonstrably wrong for the
   one clip we have (see [Calibration](#calibration)). Every metric range, lane
   width and lateral offset inherits that error. Nothing forces a calibration.
3. **The control loop has only ever been closed against a point-mass plant** with
   no grade, no drag model, no actuator dead time and no tyre limit. "No simulated
   collision" means no collision in that plant.
4. **Every threshold is an engineering default, not a calibrated vehicle
   parameter** — `k_distance`, `k_speed`, `ttc_brake_s`, `reaction_time_s`, and
   above all `brake_authority_mps2 = 8.0`, which converts a required deceleration
   into a pedal fraction. If the real vehicle delivers less at `brake = 1.0` the
   arbiter under-brakes by exactly that ratio and no software can detect it.
5. **The arbiter is not input-diverse for ego speed.** It runs its own range
   filter, lead selection and kinematics, but it consumes the same `EgoState` the
   planner does. A wrong ego speed fools both channels identically. It does at
   least reject implausible and stale states.
6. **The depth cross-check is off by default and is not a second calibrated
   sensor.** Its absolute scale is borrowed from the same camera homography, so a
   wrong calibration makes both channels wrong by the same factor. What is
   independent is the per-object measurement — whether an object's surface sits
   where its box implies, relative to the road and to other objects in the frame.
7. **The flat-road assumption is unbounded.** On a crest, a dip or a banked curve
   the road-plane homography has no valid answer and will confidently return a
   wrong number. There is no road-slope estimator and no gate on one.
8. **Nobody has validated accuracy on real driver-facing data.** The engine
   contracts are verified for shape and range semantics; sign and axis
   conventions, detector recall and the 0.35/0.50 thresholds have not been tuned
   on anything but one flat-highway clip. YOLOX-Nano finds noticeably fewer
   objects than YOLOv5n in the first ~120 frames of that clip.
9. **The systemd unit has never been started** and the Jetson container image has
   never been built. Both are reviewed, not tested.
10. **The ROS 2 bridge has never been executed.** `rclpy` is not installed here.
11. **`confirm_hits = 3` costs up to 150 ms of latency** before a genuinely new
    obstacle reaches the planner (~2 m at 15 m/s closing). That is the deliberate
    price of killing the phantom-braking path; it must be a conscious decision by
    whoever owns the safety case.
12. **`recovery_frames` and `disengage_after_frames` are frame counts, not
    times**, so they mean different durations at a frame rate other than 20 Hz.
13. **UFLD-v2 needs 413 MB in one contiguous allocation** and will fail to load
    under memory pressure. The load order mitigates it; it does not remove it.
14. **No ISO 26262 work has been done.** No HIL/SIL validation, no redundancy
    analysis, no hazard analysis, no certification.

---

## Python API

```python
import time
from adas.cli import build_pipeline
from adas.core.config import default_config
from adas.core.models import EgoState, PerceptionFrame
from adas.runtime import synthetic_frame

config = default_config()
config.detector.backend = "mock"
config.lane.backend = "mock"
config.allow_mock = True          # fabricating backends are always opt-in
config.__post_init__()            # re-validate after any override

pipeline, config = build_pipeline(config=config)
try:
    payload = synthetic_frame()
    frame = PerceptionFrame(
        frame_id=0, timestamp_s=time.monotonic(), rgb=payload,
        width=payload["width"], height=payload["height"],
    )
    ego = EgoState(speed_mps=15.0, valid=True, timestamp_s=frame.timestamp_s)

    plan, command = pipeline.step(frame, ego=ego, dt_s=0.05)

    print(plan.reason, pipeline.last_arbitration.state.value)
    print(command)                       # this is the ARBITRATED command
    print(pipeline.last_arbitration.violations)
finally:
    pipeline.close()
```

`command` is what the arbiter allowed. If you want to know what the planner
asked for and why it was changed, read `pipeline.last_arbitration`.

---

## Testing

```bash
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m pytest tests/ -q"
```

**743 tests, all passing** on this board (2026-09-13). Tests marked `engine`
exercise a real TensorRT engine and skip when the file is absent, so the suite is
green on a laptop too. Coverage is not uniform: `adas.io`, `core.metrics` and
`core.logger` are gated at 80% in CI; `runtime/capture.py`, `ros2/` and
`tools/` are covered by `tests/test_integration.py` but thinly.

---

## Repository layout

```
src/adas/core        configuration, models, validation, logging, metrics
src/adas/perception  detection, lane, camera geometry, depth, the factory
src/adas/tracking    Kalman filters, Hungarian association, the tracker
src/adas/planning    longitudinal (ACC/AEB) and lateral (LKA) laws
src/adas/control     the controller and the authoritative safety arbiter
src/adas/runtime     frame sources, ego sources, the frame loop, the pipeline
src/adas/io          Prometheus registry, health HTTP, event log, sd_notify
src/adas/ros2        OPTIONAL ROS 2 bridge (never executed on this board)
src/adas/tools       record / replay
scripts/             model fetch and TensorRT engine build
deploy/              systemd unit, logrotate, installer, Jetson image
```

## Documentation

- [docs/JETSON.md](docs/JETSON.md) — **read this first on this board**
- [ARCHITECTURE.md](ARCHITECTURE.md) — components, data flow, threading
- [QUICKSTART.md](QUICKSTART.md) — commands
- [DEPLOYMENT.md](DEPLOYMENT.md) — containers and the systemd service
- [deploy/README.md](deploy/README.md) — the operational surface in detail
- [models/README.md](models/README.md) — every model, its licence and its contract
- [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
- [docs/ROS2_INTEGRATION.md](docs/ROS2_INTEGRATION.md)
- [docs/TOOLS_GUIDE.md](docs/TOOLS_GUIDE.md)
- [CHANGELOG.md](CHANGELOG.md)

## License

MIT for the source; see [Licensing](#licensing) for the models. See `LICENSE`.

---

**Version:** 0.3.0 · **Last verified on hardware:** 2026-09-13, Jetson Xavier NX,
JetPack 5.1.6, TensorRT 8.5.2.2, Python 3.8.10
