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

**Version 0.3.0**, plus an unreleased safety-hardening pass — see
[CHANGELOG.md](CHANGELOG.md#unreleased--2026-09-13-safety-hardening-pass).

Re-measured on this Xavier NX on **2026-09-13 at 16:47 KST**, GPU mutex held, board
shared with other work (loadavg 2.9): 200 frames of `example.mp4` through YOLOX-Nano
+ UFLD-v2 + tracker + planner + controller + arbiter at **16.0 FPS end to end**
(61.9 ms/frame), 0 perception failures. Test suite: **859 passed, 1 failed** — the
failure is named and explained under [Testing](#testing). Every number in this
document was re-taken on that date; nothing is carried over.

---

## What actually works

| Capability | Backend | State |
|---|---|---|
| Object detection | YOLOX-Nano 416 (TensorRT FP16) | **Real.** Decode verified bit-identical to YOLOX's own postprocess. Default classes: person, bicycle, car, motorcycle, bus, train, truck |
| Object detection (alt) | YOLOX-Tiny 416, YOLOv5n 640 | **Real.** YOLOv5n is AGPL — see [Licensing](#licensing) |
| Lane perception | UFLD-v2 CULane ResNet-18 (TensorRT FP16) | **Real.** Canonical preprocessing, per-dataset anchors, ego boundaries by index |
| Lane + free space | YOLOP (TensorRT FP16) | **Real.** Lane fit is noisier than UFLD; the drivable-area mask is good. Must be scheduled at 2–5 Hz |
| Lane (TwinLiteNet) | — | **Honest stub.** No engine exists on this board; the decoder is written, the constructor refuses to run without weights, and the stub reports `is_mock=True` and never invents a lane |
| Relative depth (ordinal only) | MiDaS v2.1 small (TensorRT FP16) | **Loaded but demoted; off by default.** It publishes **no metric range at all**: `update()` returns `RangeSource.UNAVAILABLE`, confidence 0, for every box. The engine runs and the road-plane affine fit is good, but the measured per-object rank correlation against the reference range is 0.14–0.25 against the 0.80 bar the channel now audits itself against, so metres are gated shut and only a unitless ordinal signal (`ordinal_readings()`) is exposed. The full measurement table is the `adas.perception.depth` module docstring |
| Camera geometry | Pinhole + flat-ground homography | **Real, UNCALIBRATED by default.** Every derived metric value is confidence-capped and labelled |
| Tracking | Kalman range filter + Hungarian association | **Real.** M-of-N confirmation, class-keyed height priors, truncation flags, per-track TTC |
| Longitudinal | Constant time-gap ACC + separate AEB stage | **Real.** An AEB decision now publishes a 0 m/s target on the frame it fires, with no downward rate limit, and the controller has an emergency feed-forward — previously the planner's AEB decision could not be executed by the control path at all |
| Lateral | Speed-scheduled pixel law; Stanley law behind a calibrated camera | **Real (non-metric path active)**. The metric law refuses to engage on an uncalibrated camera |
| Safety arbitration | Independent arbiter with its own lead selection, range filter and kinematics | **Real, and authoritative.** Its in-path corridor is anchored on the image centre and never reads the tracker's `in_ego_lane` or the lane model's centre (see [Safety architecture](#safety-architecture)). It is a second opinion on the *decision*, not a second sensor |
| Operations | Prometheus `/metrics`, `/healthz`, `/readyz`, JSONL event log, systemd notify + watchdog | **Real and wired** |
| ROS 2 bridge | `adas.ros2` | **Written, NEVER EXECUTED.** `rclpy` is not installed on this board |
| Mock detector / mock lane | — | **Honest stubs.** `is_mock=True`, loud banners, refused unless `--allow-mock` |

### Measured numbers

All on the Xavier NX (MODE_20W_6CORE), **2026-09-13, 16:47–16:55 KST**, GPU mutex
held, board shared with other agents (loadavg 2.5–3.0 throughout). Every run is
`--fps 0` (unpaced), `--no-health --no-events`, source
`Ultra-Fast-Lane-Detection-v2/example.mp4`.

**Backend matrix — one command per row, `--ego-source simulated --ego-speed 15`:**

| detector + lane | frames | measured FPS | ms/frame | detections | lane rate | perception fails |
|---|---:|---:|---:|---:|---:|---:|
| YOLOX-Nano + UFLD-v2 | 200 | **16.01** | 61.9 | 136 | 100.0% | 0 |
| YOLOX-Nano + UFLD-v2 + MiDaS | 100 | 16.81 | 59.1 | 0 | 100.0% | 0 |
| YOLOX-Nano + YOLOP (auto `every_n_frames=4`) | 100 | 26.42 | 37.4 | 0 | 100.0% | 0 |
| YOLOv5n + UFLD-v2 (AGPL, dev only) | 100 | 13.71 | 72.4 | 1 | 100.0% | 0 |
| YOLOX-Nano + TwinLiteNet | — | — | — | — | — | refuses to start: no engine |

The `detections=0` rows are not a fault: over the **first 100 frames** of this clip
YOLOX-Nano finds essentially nothing, and the 136 detections of the 200-frame run all
arrive in its second half. That is a property of the clip and of YOLOX-Nano's recall,
and it is the same effect recorded in [docs/JETSON.md](docs/JETSON.md) item 8. Do not
read the 100-frame rows as a detector comparison.

**Per-stage latency, the 200-frame YOLOX-Nano + UFLD-v2 run (ms):**

| stage | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| detect (YOLOX-Nano 416) | 12.68 | 12.37 | 15.32 | 60.82 |
| lane (UFLD-v2) | 41.17 | 38.73 | 46.82 | 148.07 |
| track | 0.95 | 0.11 | 3.04 | 4.53 |
| plan | 0.21 | 0.18 | 0.29 | 1.29 |
| control | 0.08 | 0.07 | 0.10 | 0.38 |
| arbitrate | 0.46 | 0.29 | 0.97 | 2.57 |
| **frame interval** | **61.38** | **58.56** | **75.06** | **174.33** |

The `max` column on the first two rows is engine warm-up on frame 0. The decision
layer (track + plan + control + arbitrate) costs **1.70 ms/frame mean**; the budget is
still entirely perception, and within perception it is still UFLD-v2's host-side
pre/post-processing — 16.5 ms of that 41.2 ms is GPU compute per `trtexec`.

**What the arbiter did in that run, and why it is not `nominal`.** With
`--ego-source simulated` the plant closes on a lead the tracker puts 7–13 m ahead at
15 m/s, so the run spends most of its second half in `min_risk_maneuver` with
`cmd=t0.00/b0.44`. Over 200 frames: **58 safety violations, 15 warnings**, the
commonest being `jerk_*_above_4.0` (the point-mass plant's own step response) and
`range_source_switch_*` (the fused/pinhole channel alternating on one track). That is
the arbiter working, not a defect — but it means **`safety=nominal` is not the steady
state of the shipped bench command**, and any claim that it is should be distrusted.

**Mock backends (no GPU):** see [Testing](#testing) — the mock path is a CPU-only
smoke test, not a performance claim about anything the vehicle would run.

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

**How clean a recorded channel actually is, measured twice on 2026-09-13.** With a
0.5 s-spaced speed CSV and YOLOX-Nano:

| frames | detections | safety violations | warnings |
|---:|---:|---:|---:|
| 50 | 0 | **0** | **0** |
| 400 | 456 | **303** | 13 |

The 50-frame result is the one this README used to quote on its own, and it does
reproduce — `safety=nominal` on every frame, throttle rising 0.00 → 0.09 as the
controller closes on cruise. But it is 50 frames of clip in which the detector finds
**nothing at all**, so it measures an empty road, not a clean system. Over the full
400 frames, where 456 detections and 493 tracks appear, the arbiter intervenes on the
great majority of frames — mostly `jerk_*_above_4.0` from the plant's own step
response and `range_source_switch_*` on a single track. Quote the 400-frame row.

---

## Safety architecture

```
capture → detect → lane → track → depth cross-check → plan → control → ARBITRATE → actuators
                                                                          ▲
                                        the arbiter's command is the only one that leaves
```

The arbiter (`adas.control.arbiter.SafetyArbiter`) is deliberately independent of
the planner. What that independence actually consists of, as the code stands today:

* **Its own lead selection**, over the raw track list, ordered by TTC then range.
  Its in-path corridor is anchored on the **image centre** with a half width of at
  least `ArbiterLimits.min_in_path_half_width_frac` (0.30 of the frame width), and
  membership is by box overlap. It does **not** read `TrackedObject.in_ego_lane`
  (the tracker computes that from the same lane model the planner reads) and a lane
  model may only *widen* the corridor, never move or narrow it, and only when the
  lane is not mock, its centre is finite and inside the frame, and its confidence is
  at least `lane_trust_confidence` (0.50). A bad lane centre can therefore no longer
  hide a real lead from the arbiter.
* **Its own alpha-beta range filter**, seeded on a new or re-initialised track from
  `-max(0, ego_speed)` — the safe prior that an unknown object is stationary in the
  world. It does not read `TrackedObject.velocity_mps`.
* **Its own kinematics** differenced from measured ego speed, and its own RSS
  minimum-gap test.
* **Its own output invariant**: the returned command is never more energetic than the
  input. Throttle is only ever reduced; **brake is only ever increased**, enforced
  structurally by a final `max(brake, cmd_in.brake)`. There is deliberately no brake
  *apply*-rate limit in the arbiter — brake application jerk belongs to the
  controller, which owns the emergency exemption. Only the throttle apply-rate and
  the brake *release*-rate floor survive as output shaping.

**What is NOT independent, and matters.** `SafetyContext.tracks` is the tracker's
output, so a detection perception never produced is invisible to the arbiter too, and
`EgoState` is the same object the planner reads, so a wrong ego speed fools both
channels identically. The arbiter is a second opinion on the **decision**, not a
second sensor.

Three properties are enforced by tests in `tests/test_pipeline.py` and
`tests/test_arbiter.py`:

* **A runaway controller cannot actuate.**
  `test_arbiter_command_is_what_the_pipeline_returns` installs a controller that
  always commands full throttle and asserts the returned throttle is 0.
* **A perception failure is a fault, not an empty road.** An exception from the
  detector or lane estimator produces `PerceptionStatus(ok=False)` with a rising
  `consecutive_failures`, and the planner is told `perception_valid=False`. No
  detection is fabricated and no track is corrected, but the tracker **is** advanced
  by a predict-only step (`ADASPipeline._coast_tracks`): live tracks coast with
  growing covariance and rising `time_since_update`, their range estimate becomes
  `RangeSource.UNAVAILABLE`, and they are deleted at `max_missed`. Freezing them —
  which is what the previous build did — made the recovery frame associate a real
  measurement against an N-frame-stale prediction that still claimed full confidence.
  An empty detection list from a *working* detector still means the road is clear.
* **Leaving the frame loop is a command too.** `PipelineRunner` classifies every loop
  exit. On a *clean* exit (EOF, frame budget, operator stop) it emits exactly one
  zero-throttle, zero-brake release. On an **unsafe** exit — `source_lost` or
  `pipeline_dead` — it re-emits `ADASPipeline.failsafe_command()` once per nominal
  period for `failsafe_hold_s` (default 1.0 s, i.e. 20 commands at 20 Hz) so the
  arbiter's brake ramp actually reaches the minimum-risk manoeuvre. The hold is
  **bounded**, not indefinite: it hands back to the caller, and `deploy/adas.service`
  is the thing that restarts the process. Previously a mid-stream sensor loss took
  `break` and left the last throttle latched on the actuators for ever.

When `pipeline.step` raises, the runner actuates
`ADASPipeline.failsafe_command()` and counts the failure; ten consecutive failures
stop the loop. It never latches the previous command on the actuators.

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
  "depth":    { "backend": "off", "cadence_frames": 5 },
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
height and pitch. It is the switch that tells the whole stack its metres are real.

`depth` stays `off`. Turning it to `"midas"` costs an engine load and buys nothing
today: the channel publishes no metric range (see the model table), so the arbiter
runs on the pinhole channel alone either way. Measured on this board, adding
`--depth midas` to the reference run changed the frame rate from 16.0 to 16.8 FPS —
i.e. within the board's own run-to-run noise — and produced zero
`range_channel_*` findings, because there is nothing for the fusion to consume.

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
  `adas_stage_duration_ms{stage}`, `adas_lane_is_mock`, `adas_ego_speed_valid`,
  `adas_ram_mb` and `adas_build_info`.

Two fields on `/healthz` are new and are worth knowing about:

* **`ram_mb`** is now populated on every probe and every scrape (re-read from
  `/proc/self/statm`, rate-limited to once a second). It used to be `null` always,
  because nothing called `HealthState.refresh_process()`, while `adas_ram_mb` scraped
  as the literal `0` for a process holding over a gigabyte. When RAM genuinely cannot
  be measured the gauge now exports `NaN`, never `0`.
* **`engine_sha256`** maps engine filename → the sha256 this process actually
  verified. Engines are verified against `models/MANIFEST.json` **before** they are
  deserialised, and a digest mismatch or a manifest/runtime TensorRT version mismatch
  refuses the load (`EngineIntegrityError`). An engine the manifest does not list
  logs a WARNING and loads as unverifiable. There is no environment-variable bypass.
  Copying an engine in by hand without updating the manifest will now stop the
  process, by design.

The event log (`data/events.jsonl` in a lab profile) records lifecycle,
safety-state transitions, perception dropouts, engine failures and source events
as JSON lines. Entering `MIN_RISK_MANEUVER` or `DISENGAGE` is CRITICAL and is
`fsync`ed immediately.

**The event schema is now v2, and the ordering rules changed.** Sort by `seq`: it
continues from the last record already in the file at startup, so it is the
authoritative order of a file *across a restart*. `t` is stamped unconditionally from
`time.monotonic()` inside the writer — one field, one clock domain, never
caller-supplied — and is comparable only within one `boot` id, because the kernel
monotonic clock restarts at zero at boot. A previous version of this documentation
claimed `boot` + `t` gave "a total order across restarts"; that was false and has been
removed. `ts` (wall clock) can step backwards at an NTP correction. A new 12-hex-digit
`run` field distinguishes two writer instances sharing one file. No field orders
records written into *different* files.

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
5. **The arbiter is not input-diverse for ego speed, nor for perception.** It runs
   its own range filter, lead selection, in-path corridor and kinematics, but it
   consumes the same `EgoState` the planner does and the same tracker output. A wrong
   ego speed fools both channels identically, and a detection perception never
   produced is invisible to both. It does at least reject implausible and stale
   states. It is a second opinion on the decision, not a second sensor.
6. **There is no independent range channel. The depth cross-check has been
   demoted.** MiDaS v2.1 small was documented here as an independent metric range
   cross-check; it is not one and it never was. Measured on this board over the
   reference clip, its per-object rank correlation with the reference range is
   **0.14–0.25** (best variant found: 0.686, still short of the 0.80 bar and still
   collapsing the far field), it compressed a true 5.3–61.5 m spread into 3.7–13.4 m,
   and at frame 300 it reported the 54 m car *nearer* than the 13 m car. The channel
   now publishes `RangeSource.UNAVAILABLE` with confidence 0 for every box by
   default; `publish_metric=True` is a *request* that only takes effect if a rolling
   self-audit measures a Spearman correlation of at least 0.80 over at least 24 pairs,
   and it fails closed. So: **the arbiter is running on the pinhole channel alone**,
   and the "two-channel cross-check" in the architecture diagram is currently one
   channel plus a gate. The road-plane affine fit still works and is still reported
   in `stats()`; it is the per-object sampling that carries no range information.
7. **An uncalibrated camera raises no violation in a live run.** The arbiter has the
   mechanism — `SafetyContext.camera_calibrated=False` appends `camera_uncalibrated`,
   floors the state at LIMITED and cuts throttle unless
   `safety.allow_uncalibrated_range` is set — and it is unit-tested. But
   `src/adas/runtime/pipeline.py` does not pass `camera.calibrated` into the
   `SafetyContext`, so on this board today a run with the shipped
   `CameraConfig(calibrated=False)` still reports `safety=nominal` while acting on
   assumed metric ranges. One line in `pipeline.py` turns it on.
8. **The flat-road assumption is unbounded.** On a crest, a dip or a banked curve
   the road-plane homography has no valid answer and will confidently return a
   wrong number. There is no road-slope estimator and no gate on one.
9. **Nobody has validated accuracy on real driver-facing data.** The engine
   contracts are verified for shape and range semantics; sign and axis
   conventions, detector recall and the 0.35/0.50 thresholds have not been tuned
   on anything but one flat-highway clip. YOLOX-Nano finds noticeably fewer
   objects than YOLOv5n in the first ~120 frames of that clip.
10. **The systemd unit has never been started** and the Jetson container image has
   never been built. Both are reviewed, not tested.
11. **The ROS 2 bridge has never been executed.** `rclpy` is not installed here.
12. **`confirm_hits = 3` costs up to 150 ms of latency** before a genuinely new
    obstacle reaches the planner (~2 m at 15 m/s closing). That is the deliberate
    price of killing the phantom-braking path; it must be a conscious decision by
    whoever owns the safety case.
13. **`recovery_frames` and `disengage_after_frames` are frame counts, not
    times**, so they mean different durations at a frame rate other than 20 Hz.
14. **UFLD-v2 needs 413 MB in one contiguous allocation** and will fail to load
    under memory pressure. The load order mitigates it; it does not remove it.
15. **No ISO 26262 work has been done.** No HIL/SIL validation, no redundancy
    analysis, no hazard analysis, no certification.
16. **No soak test exists.** The longest run in this repository's history is a few
    hundred frames — about 25 seconds of clip. There is no 8 h result, no RSS curve
    over time, no file-descriptor audit and no evidence about what the alpha-beta
    filters, the event log rotation or the tracker's id space do after an hour.
17. **One test fails and is left failing.**
    `tests/test_integration.py::test_record_and_replay_round_trip` asserts a recorded
    arbitration brake of 1.0 against a replayed 0.0. The cause is in
    `src/adas/tools/replayer.py`: `RecordedDetector.infer` looks up
    `getattr(frame, "frame_id")`, but `ADASPipeline.step` hands it `frame.rgb` — a raw
    image, which has no `frame_id` — so **every replayed frame returns zero
    detections** and the replay re-runs planning, control and arbitration on an empty
    road. The test only passed before because the recorded side also happened to
    command brake 0. The replay tooling is therefore not a usable regression harness
    until that is fixed.
18. **The arbiter logs one WARNING per frame** in a steady degraded state (no ego
    speed, or a persistent hazard). A 200-frame bench run produces ~200 arbiter
    WARNING lines. The planner and lateral/behaviour layers were fixed with a
    `LogGate` (entry, one repeat per period with a suppressed count, one exit line);
    `src/adas/control/arbiter.py` has not been converted and `grep -c LogGate` on it
    returns 0. At 20 Hz that is 72,000 lines an hour into the journal.

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

**860 tests collected: 859 passed, 1 failed**, measured on this board on 2026-09-13
in 50.75 s. The single failure is
`tests/test_integration.py::test_record_and_replay_round_trip` (`assert 1.0 == 0.0 ±
1e-06`), and it is a real defect in `src/adas/tools/replayer.py`, not a flake — see
item 17 of [What is not production ready](#what-is-not-production-ready). It is left
failing deliberately rather than deleted or skipped.

`pyproject.toml` no longer sets `addopts = "-q"`. It used to, and because the
documented command *also* passes `-q`, pytest read `-q -q` as `-qq` and suppressed the
pass/fail summary entirely — the project's own test command printed no counts. The
verbosity flag now comes from the invocation.

Tests marked `engine` exercise a real TensorRT engine and skip when the file is
absent, so the suite runs on a laptop too. Coverage is not uniform: `adas.io`,
`core.metrics` and `core.logger` are gated at 80% in CI; `runtime/capture.py`, `ros2/`
and `tools/` are covered by `tests/test_integration.py` but thinly. And coverage is
not correctness: **nothing in this repository has been validated against ground
truth.**

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

**Version:** 0.3.0 (plus an unreleased hardening pass) ·
**Last verified on hardware:** 2026-09-13 16:47–17:00 KST, Jetson Xavier NX,
JetPack 5.1.6, TensorRT 8.5.2.2, Python 3.8.10, board shared (loadavg 2.6–4.0).

Every number in this file was taken in that window by running the commands written
here. Where a claim could not be reproduced it was corrected or deleted; where
something has not been measured, the text says so.
