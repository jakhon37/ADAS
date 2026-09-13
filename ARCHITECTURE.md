# Architecture

A camera-only ADAS pipeline for the Jetson Xavier NX. This document describes
what the code does, including where it deliberately refuses to produce an
answer. For what is *not* ready, see
[README.md](README.md#what-is-not-production-ready).

## Data flow

```
                    adas.runtime.capture
  ┌──────────────┐  ┌──────────────┐
  │ FrameSource  │  │ EgoSpeed     │   there is no vehicle bus on this board;
  │ video/csi/   │  │ Source       │   the ego source states whether its number
  │ synthetic    │  │              │   is a measurement (README: Ego speed)
  └──────┬───────┘  └──────┬───────┘
         │ CapturedFrame   │ EgoState(valid, ...)
         ▼                 ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ adas.runtime.runner.PipelineRunner                                   │
  │  measured dt (clamped 0.5x..3x nominal) · pacing · reconnect ·       │
  │  fail-safe on a failed frame · cooperative stop · health/events/wdog │
  └──────────────────────────────┬───────────────────────────────────────┘
                                 │ PerceptionFrame + EgoState + dt_s
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ adas.runtime.pipeline.ADASPipeline.step                              │
  │                                                                      │
  │  detect ──► lane ──► track ──► depth cross-check ──► plan ──► control│
  │    │          │        │             │                 │        │    │
  │    └──────────┴────────┴─ PerceptionStatus(ok, consecutive_failures) │
  │                                                          │           │
  │                                                          ▼           │
  │                                           ┌──────────────────────┐   │
  │                                           │  SafetyArbiter       │   │
  │                                           │  AUTHORITATIVE       │   │
  │                                           └──────────┬───────────┘   │
  └──────────────────────────────────────────────────────┼───────────────┘
                                                         │ ArbitrationResult
                                                         ▼
                                              actuators / ROS 2 bridge
```

The arrow out of the arbiter is the only one that reaches an actuator. Nothing
downstream ever sees the controller's un-arbitrated request except through
`ADASPipeline.last_arbitration`, which carries it for logging and diagnostics.

## Components

### `adas.core` — configuration, models, validation, logging, metrics

No numpy, no OpenCV, no TensorRT: this package imports on any machine and is
what the ops layer and most tests build against.

* **`config.py`** — one dataclass per subsystem, every field range-checked at
  construction. `load_config` rejects unknown keys (a typo in a safety limit used
  to be silently dropped), accepts a short explicit list of deprecated keys with
  a warning, and cross-validates sections against each other:
  planner ⊆ safety for speed, steering, acceleration and following distance, and
  `safety.max_road_wheel_rad == radians(controller.max_steering_angle_deg)`
  (leave it `0.0` to derive). `RuntimeConfig.allow_mock` gates every fabricating
  backend.
* **`models.py`** *(owned by the orchestrator)* — `PerceptionFrame`,
  `TrackedObject`, `LaneModel`, `EgoState`, `PerceptionStatus`, `RangeEstimate`,
  `SafetyState`, `ArbitrationResult`.
* **`validation.py`** — raises `ValidationError`; never repairs. Notably
  `validate_control_command` now checks brake (it did not), bounds throttle to
  `[0, 1]` (it allowed `[-1, 1]`), bounds steering, and rejects throttle and
  brake together.
* **`logger.py`** — one root handler, installed once by `configure_logging`;
  text or JSON; a closed event vocabulary; `Throttle` for per-frame chatter.
* **`metrics.py`** — measured FPS from `time.monotonic` (not from the caller's
  claimed frame period), a bounded latency ring with exact p50/p95/p99, and a
  `metrics.stage("detect")` context manager that records in a `finally`.

### `adas.perception` — detection, lane, geometry, depth

* **`factory.py`** is the only place a configuration string becomes a loaded
  engine. A missing engine is an error; substituting a mock needs
  `allow_fallback`, which the CLI sets only when the operator passed
  `--allow-mock`. Model paths resolve against the CWD *and* the repository root.
  It loads a large engine only after warning if available memory is under 1.6×
  the file size.
* **`detection.py`** — layout-independent primitives: preprocessing contracts per
  model family, letterbox transforms with a vectorised inverse, class-aware NMS,
  and `analyse_detection_head`, which *derives and validates* the head layout
  from the engine's declared shape rather than guessing.
* **`yolo.py` / `yolox.py`** — `YoloTensorRTDetector` reads the engine's contract
  at construction; `YoloXTensorRTDetector` pins the YOLOX layout. Per-stage
  timing is exposed (`last_preproc_ms`, `last_infer_ms`, `last_decode_ms`).
* **`ufld.py`** — UFLD-v2 with the canonical preprocessing (stretch to 1600×533,
  ImageNet normalisation, keep the bottom 320 rows), per-dataset anchors, output
  tensors bound **by name**, and ego boundaries taken by index (1 = left,
  2 = right) rather than by position.
* **`yolop.py`** — lane lines, drivable area and vehicle detections from one
  pass. ~59 ms end to end: it must be scheduled, and it deliberately does no
  internal frame skipping, because re-serving a stale lane as current would be
  dishonest. The pipeline owns the schedule (below).
* **`twinlite.py`** — the decoder is written; no engine exists on this board.
  `TwinLiteNetUnavailable` is the honest stub: `estimate() -> None`,
  `is_mock=True`, drivable area confidence 0.
* **`geometry.py`** — `CameraConfig` with a road-plane homography built once,
  `image_to_ground` / `ground_to_image` returning `None` above the horizon,
  per-class pinhole range, ground-plane range from the contact point, confidence-
  weighted fusion in inverse range, truncation policy, metric lane fitting, and
  `LaneCalibrator` for monocular self-calibration. `calibrated=False` caps every
  derived confidence at 0.6 and is the default.
* **`depth.py`** — MiDaS as an *independent* range channel. The affine scale is
  fitted against road-plane anchors projected through the homography, so the
  object range shares no pixels with the box-height estimate. Reduced cadence
  (default every 5 frames) with linearly decaying confidence and a hard expiry.
  No engine ⇒ honest stub, everything `RangeSource.UNAVAILABLE`.

### `adas.tracking` — Kalman + Hungarian

`RangeFilter` is constant-*acceleration* over `[d, ḋ, d̈]` in metres: a CV filter
lags by `a·τ` through the whole of a lead-braking event, which is exactly when
TTC must be right. `BoxFilter` is CV on the image-plane centre plus two scalar
random walks on width and height (a size *rate* diverges during a coast).
Association is a global Jonker–Volgenant solve over a cost matrix with hard
vetoes (class group, absolute distance, size ratio, chi-square gate) — no greedy
nearest-centre, so nothing is track-creation-order dependent. Tracks are
TENTATIVE until M-of-N confirmation (3 of 5); `update()` returns confirmed tracks
only. `in_ego_lane` is a four-tier decision (metric lane corridor → drivable
corridor → metric ego corridor → range-scaled image band) and `diagnostics()`
records which tier decided each track.

### `adas.planning` — longitudinal and lateral

`LongitudinalPlanner` is one continuous constant-time-gap law with no branches
(`d_desired = d₀ + T·v`, `v = clamp(v + k_d·(d − d_desired) + k_v·v_rel, 0,
cruise)`) plus a separate AEB stage on TTC, required deceleration and the
standstill gap. A stateful rate limiter bounds how fast the target may fall or
rise; a coasting lead blocks any increase. An invalid ego speed degrades to
hold-and-ramp-down and reports `ego_speed_unavailable` — it never guesses.

`LateralPlanner` has two laws behind one interface. The non-metric law
(shipping) is speed-scheduled proportional control on the normalised pixel error
and is explicitly labelled `is_metric=False` with `lateral_error_m=None`. The
metric Stanley law needs a calibrated camera and
`CameraGeometry.from_camera_config` refuses an uncalibrated one, because an
assumed mount height would turn `lateral_error_m` into a fabricated measurement
wearing metric units. Both are capped by `δ_max = atan(a_lat_max·L/v²)`.

### `adas.control` — controller and arbiter

`PIDLikeLongitudinalController` is stateful: deadband with hysteresis on actuator
selection → PI producing an **acceleration** (gains are 1/s and 1/s²) with
clamping anti-windup → jerk limit → pedal map through the authority constants →
pedal rate limits. It validates its own output.

`SafetyArbiter.arbitrate(plan, command, SafetyContext) -> ArbitrationResult` is
the authority. It is independent of the planner *by construction*:

| quantity | planner | arbiter |
|---|---|---|
| lead selection | nearest in-lane | smallest TTC, then smallest range, over the RAW track list |
| range rate | tracker's Kalman | its own alpha-beta filter with a jump gate |
| in-path test | `in_ego_lane`, else lane centre | `in_ego_lane`, else its own wider image band |
| kinematics | plan over a horizon | differenced from measured ego speed |
| headway rule | time gap | RSS minimum gap |

It also cross-checks range against a second channel when one is supplied
(`SafetyContext.independent_ranges`), degrades on >30% disagreement and uses the
nearer value. State machine: NOMINAL → LIMITED → MIN_RISK_MANEUVER → DISENGAGE,
escalation immediate, de-escalation only after `recovery_frames` clean frames,
DISENGAGE latched until `reset()`. Only *faults* accumulate toward DISENGAGE;
*hazards* are traffic the arbiter exists to handle, so a long approach to a
stopped queue keeps braking rather than disengaging.

The legacy raise-based `SafetyMonitor` methods still exist and are documented in
code as **ADVISORY**. Nothing on the actuation path calls them any more; the
pipeline calls `arbitrate` only.

### `adas.runtime` — sources, loop, pipeline

**`capture.py`** owns where frames and ego speed come from. `FrameSource.eof`
distinguishes "the clip ended" from "the device stopped delivering":
`ReconnectingSource` retries only the second. `EgoSpeedSource` has four
implementations, each declaring `measured`.

**`pipeline.py`** runs one frame. Three behaviours are the point of the module:

1. The returned command is the arbiter's.
2. A perception exception sets `PerceptionStatus(ok=False)`, does **not** advance
   the tracker (an empty detection list would read as "every object vanished"),
   and tells the planner `perception_valid=False`. A detector fault and a lane
   fault are tracked separately. An empty list from a working detector still
   means the road is clear.
3. Lane scheduling is honest. With `lane.every_n_frames > 1`, a reused model is
   republished with its confidence scaled by `1 − age/(max_age+1)` and dropped
   entirely past `lane.max_age_frames`. Old evidence is labelled as old evidence.

The depth channel is fed the *track* boxes, not the raw detections, so its
results are index-aligned with the tracks and need no matching heuristic;
`UNAVAILABLE` results are dropped rather than published as a zero-confidence
range, because a missing key means "no second opinion", which is a different
claim from "the second opinion is 0 m".

**`runner.py`** owns timing and failure policy: measured `dt` clamped to
`[0.5×, 3×]` nominal, `--frames 0` continuous mode, `request_stop()` for signals,
and — when `step` raises — `pipeline.failsafe_command()` on the actuators plus a
consecutive-failure counter that stops the loop, never a `continue` that latches
the last command.

### `adas.io` — operations

A stdlib Prometheus registry (no pip dependency), a `ThreadingHTTPServer` on a
daemon thread serving `/livez` `/readyz` `/healthz` `/metrics`, an append-only
JSONL event log with two rotation mechanisms and a disk-full state machine, and
`sd_notify` with a `WatchdogPinger` that pings **only when the frame id has
advanced** — a wedged pipeline stops petting the watchdog and systemd kills it.

### `adas.ros2` — optional, never executed

Importable without `rclpy`; `ADASBridgeNode` raises `ImportError` at construction
rather than degrading into a node that silently publishes nothing. Speed messages
are validated and time-stamped, and the ego state goes invalid after
`speed_timeout_s`; a failed frame publishes the fail-safe command instead of
latching; diagnostics carry the arbiter's real state.

## Threading

There are no stateless components. Every backend owns a TensorRT execution
context and a set of pinned host buffers; the tracker, the planner's rate
limiter, the controller's integrator, the arbiter's filters and latches, and
`PerformanceMetrics` are all single-writer.

**One pipeline instance belongs to one thread.** The only objects designed for
concurrent access are `adas.io.metrics` (registry-wide `RLock`) and
`adas.io.health.HealthState` (its own `RLock`), which the HTTP threads read
through.

## Error model

| condition | representation | consequence |
|---|---|---|
| no object detected | empty `detections` | road is clear |
| no lane visible | `lane is None`, no exception | lateral law holds |
| detector/lane raised | `PerceptionStatus(ok=False)` | planner degrades, arbiter escalates on repeats |
| no ego speed | `EgoState(valid=False)` | hold and ramp down; MRM |
| no metric lane geometry | `lateral_offset_m=None` | arbiter reports `lane_offset_unavailable` |
| no second range channel | key absent from `independent_ranges` | lead keeps `RangeSource.PINHOLE` |
| range unmeasurable (tiny box) | `RangeSource.UNAVAILABLE` | not fused, not zero |
| `step` raised | `ADASException` | runner actuates `failsafe_command()` |
| engine missing | `PerceptionError` at build | process refuses to start |

Nothing in that table is represented by a plausible default value. That is the
single design rule this codebase is organised around.

## Extension points

* **A new detector**: implement `infer(frame, width, height) -> [BoundingBox]`,
  add a branch to `perception/factory.build_detector`, add the backend name to
  `DETECTOR_BACKENDS` in `core/config.py`.
* **A new lane backend**: subclass `perception.lane.LaneBackend` (`estimate`,
  optional `drivable_area`, `is_mock`, `close`) and add it to
  `build_lane_estimator` and `LANE_BACKENDS`.
* **A second range channel**: return `RangeEstimate` objects and hand them to
  `SafetyContext.independent_ranges` keyed by track id — the arbiter's
  cross-check is already coded against it.
* **A real ego source**: subclass `runtime.capture.EgoSpeedSource`, set
  `measured = True` only if it is one, and dispatch in `open_ego_source`.
