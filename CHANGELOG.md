# Changelog

## [0.3.0] - 2026-09-13

### Real models, an authoritative safety arbiter, and an operable process

Six TensorRT engines replace one; the safety monitor stops being advisory and
becomes the thing that actuates; the pipeline stops reading a perception failure
as an empty road; and the process becomes supervisable. Measured on the target
Jetson Xavier NX: **200 frames of `example.mp4` through YOLOX-Nano + UFLD-v2 +
tracker + planner + controller + arbiter at 15.28 FPS end to end, 0 failures**.
**743 tests pass** (was 37 at 0.2.0).

Read `README.md` — specifically *What is not production ready* — before quoting
any of this. It has still never driven a vehicle, there is still no camera on
this board, and the shipped camera calibration is still an assumption.

### Safety — behaviour changes

- **`SafetyMonitor.arbitrate` is authoritative.** `ADASPipeline.step` returns the
  arbiter's command, not the controller's. The previous pipeline caught the
  safety violation, logged it at WARNING, and actuated the rejected command
  anyway. Covered by `tests/test_pipeline.py::test_arbiter_command_is_what_the_pipeline_returns`,
  which installs a controller that always commands full throttle and asserts the
  returned throttle is 0.
- **A perception failure is a fault, not an empty road.** An exception from the
  detector or lane estimator now produces `PerceptionStatus(ok=False)` with a
  rising `consecutive_failures`, does **not** advance the tracker with a
  synthetic empty detection list, and tells the planner `perception_valid=False`.
  Detector and lane faults are tracked separately. An empty list from a working
  detector still means the road is clear.
- **A failed frame actuates a fail-safe.** `PipelineRunner` calls
  `ADASPipeline.failsafe_command()` — the arbiter's minimum-risk command — and
  counts the failure; ten consecutive failures stop the loop. It previously
  `continue`d, leaving the last command latched on the actuators.
- **Ego speed is never invented.** `EgoState` is threaded through the pipeline
  with its `valid` flag. `ADASPipeline.step(current_speed_mps=...)` now defaults
  to `None`, meaning *no ego speed*, rather than `0.0`, which used to be
  indistinguishable from a measured standstill.
- **`validate_control_command` closes three holes**: it never checked `brake` at
  all (so a NaN reached a sanitiser whose `max(0, min(1, nan))` is `1.0` — a full
  ABS stop synthesised from corrupt data), it permitted throttle in `[-1, 1]`,
  and it never bounded steering. Throttle and brake commanded together is now
  rejected.
- **Mock backends are opt-in.** A `mock` detector or lane backend is refused
  unless `allow_mock` is set (`--allow-mock`, `ADAS_ALLOW_MOCK=1`, or
  `"allow_mock": true`). Substituting a mock because an engine file was missing
  needs the same permission and is logged at ERROR.

### Perception

- YOLOX-Nano 416 (Apache-2.0) is the default detector; YOLOX-Tiny and YOLOv5n are
  selectable. Default class set is road users **including pedestrians**.
- UFLD-v2 CULane ResNet-18 is the default lane backend, with the canonical
  preprocessing, per-dataset anchors and output tensors bound by name.
- YOLOP (lane + drivable area + vehicles) and TwinLiteNet backends added.
  TwinLiteNet has no engine on this board and returns an honest unavailable stub.
- MiDaS v2.1 small wired as an **independent** range cross-check, off by default,
  scale-anchored on road-plane points so it shares no pixels with the box-height
  estimate.
- Camera geometry (`perception/geometry.py`) with a road-plane homography,
  per-class pinhole range, truncation policy and monocular self-calibration.
  Intrinsics are rescaled to the live frame size (an `fx` measured at 1280 is
  1.5× too small at 1920 and nothing used to notice).
- **Load order matters**: the pipeline loads the largest engine first. UFLD-v2 is
  413 MB and needs one contiguous allocation; measured on this board,
  detector-then-lane fails with `Cuda Runtime (out of memory)` at ~1.1 GB
  available while lane-then-detector succeeds. The factory warns when available
  memory is under 1.6× the engine size.

### Tracking, planning, control

- Kalman range filter (constant acceleration) + Hungarian association replacing
  greedy nearest-centre; M-of-N confirmation, class-keyed height priors, per-track
  TTC, metric lateral offset, `in_ego_lane` with recorded provenance.
- Constant time-gap ACC with a separate AEB stage and a stateful target rate
  limiter; hold-and-ramp-down on a perception dropout or an unusable ego speed.
- Lateral law explicitly labelled metric or non-metric; the metric Stanley law
  refuses to engage on an uncalibrated camera.
- Controller is stateful: PI producing an acceleration, clamping anti-windup,
  jerk limit, pedal map, pedal rate limits.

### Runtime and configuration

- `--frames 0` runs until stopped. The loop was `for frame_id in range(max_frames)`,
  so a service launched with `--frames 0` exited immediately and restarted forever.
- `dt_s` is now **measured** between frames (clamped to `[0.5×, 3×]` nominal),
  not the scheduled period.
- SIGTERM/SIGINT stop the loop cooperatively; SIGHUP reopens the event log.
- Frame sources distinguish end-of-file from a read failure and reconnect only on
  the second; `--loop` restarts a clip.
- **Ego speed sources**: `none` (default, honest), `config`, `simulated`
  (point-mass plant driven by the actuated command), `file` (a recorded
  `timestamp_s,speed_mps` channel that ages out rather than freezing).
- **Configuration is strict.** Unknown keys are a hard error naming the section;
  a small list of deprecated lane keys is accepted with a warning; sections
  cross-validate (planner ⊆ safety for speed, steering, acceleration and
  following distance) and `safety.max_road_wheel_rad` is derived from
  `controller.max_steering_angle_deg` or must match it. New sections: `camera`,
  `ego`, `depth`, `health`, `events`. `schema_version` is checked.
- Every configuration value now actually reaches its component. Roughly half of
  the planner, controller, safety and tracker fields were previously constructed
  and then ignored.
- `ADAS_CONFIG_PATH` is read. `--log-level` is applied **before** the config is
  loaded, so it can suppress the loader's own output.
- `--print-config` validates and prints the resolved configuration.

### Operations

- `adas.io` wired into the CLI: Prometheus `/metrics`, `/healthz`, `/readyz`,
  `/livez` on `127.0.0.1:8090`; a durable JSONL event log; systemd `READY=1`
  after the **first completed frame**; a watchdog that pings only when the frame
  id has advanced.
- Per-stage latency histograms (`detect`, `lane`, `track`, `plan`, `control`,
  `arbitrate`, `depth`) recorded and exported.

### Tools

- `DataRecorder` records the arbitration result, the tracks, the ego state and
  the perception status, and no longer raises `AttributeError` on the first frame
  containing a lane (it read three `LaneModel` fields that do not exist).
- `DataReplayer` reconstructs a `LaneModel` from its real fields, replays the
  recorded `EgoState` including its validity, and exposes `safety_timeline()`.
- `replay_with_pipeline` swaps in perception backends that serve the recorded
  detections and lanes, so tracking, planning, control and arbitration can be
  re-run on any machine, GPU or not.

### ROS 2

Still **never executed** — `rclpy` is not installed on this board — but no longer
fails open where it is wrong to:

- The ego speed is validated and time-stamped and goes invalid after a timeout.
  Previously `current_speed_mps` started at `0.0` and was assigned unchecked, so a
  dead `/vehicle/speed` made the planner see a stopped vehicle and command full
  throttle forever.
- A failed frame publishes the fail-safe command instead of latching the last one.
- Diagnostics report the arbiter's real state instead of a hard-coded
  `DiagnosticStatus.OK` / "Operating normally".
- The decoded image is passed to the detector, not a `{"width", "height", "data"}`
  dict.
- `adas.ros2` imports without ROS 2 installed.

### Packaging and licensing

- `dependencies = ["numpy>=1.19"]`. The package declared none while importing
  numpy at module scope. OpenCV and TensorRT stay undeclared on purpose: on the
  Jetson both come from JetPack and a pip OpenCV shadows the GStreamer build.
- The `ros2 = ["rclpy>=3.0"]` extra is removed — it never produced a working
  install. Use apt and a ROS 2 underlay.
- `MANIFEST.in` excludes every `*.onnx`, `*.engine` and `*.mp4` and ships the
  deployment assets and `models/MANIFEST.json`.
- **AGPL-vs-MIT resolved**: YOLOv5n (AGPL-3.0-only) is no longer the default
  detector, is not distributed, and is replaced by YOLOX-Nano (Apache-2.0), which
  is both verified equivalent in decode and faster. The declared MIT licence now
  covers everything actually shipped. Dataset caveats (CULane, BDD100K:
  research/non-commercial) are documented and are *not* laundered by the MIT code
  licence.
- `ruff` lint rules are pinned; `engine` and `needs_fixture` pytest markers are
  registered.

### Documentation

`README.md`, `ARCHITECTURE.md`, `QUICKSTART.md`, `DEPLOYMENT.md` and
`docs/JETSON.md` rewritten. The previous README carried ten green production
ticks, several of which the code contradicted: "Stateless Design: Thread-safe"
(nothing in the pipeline is stateless), "Docker Support: Production-ready
containerization" (the image could not load an engine), "Debug Tools:
Record/replay" (it crashed on first use), and a Python API example importing
`adas.pipeline` and `adas.models`, neither of which exists. Those are gone,
replaced by measured numbers, the real model list with licences, and an explicit
fourteen-item *What is not production ready* list.

### Known-broken, deliberately not fixed here

- No camera on this board; the CSI path is unexecuted.
- The shipped camera calibration is an assumption and is measurably wrong for
  `example.mp4` (horizon at v = 423 px, i.e. pitch −3.87°, not the 328 px the
  shipped `pitch_deg: 2.0` implies). Nothing forces a calibration.
- The systemd unit has never been started; `deploy/Dockerfile.jetson` has never
  been built.
- No overlay or saved debug video.

---


## [0.2.0] - 2026-09-09

### Jetson-native Python 3.8 runtime

ADAS now runs as a single process on JetPack 5 (Python 3.8 + TensorRT 8.5) instead of requiring 3.10+.

- Dropped `@dataclass(slots=True)` so the package imports on CPython 3.8
- `requires-python = ">=3.8"`
- Pinhole range: `distance = (object_height_m * focal_length_px) / box_height` (defaults 1.5 m, 910 px)
- Track range-rate velocity
- Swappable perception: `mock` (default) | `tensorrt` YOLO | `ufld` lanes
- Video / CSI / OpenCV camera sources (`--source`)
- Safety plan accel/decel judged over a 1 s horizon
- `scripts/build_yolo_engine.py` waits for free RAM/GPU before `trtexec`
- Verified on Xavier NX: YOLOv5n FP16 engine, ~45 ms/frame end-to-end on `example.mp4`

### Notes

TensorRT backends stay lazy-imported. Mock tests do not load CUDA. Build engines only when the board is idle (DMS replay holds ~0.8 GB + GPU). Pickup doc: `docs/JETSON.md`.

---

## [0.1.0] - 2026-02-26

### Major Refactoring: Production-Ready ADAS Core

This release transforms the project from tutorial-style code snippets to a production-grade ADAS system with industry standards.

### 🚀 Added

#### Core Features
- **Complete modular architecture** with perception → tracking → planning → control pipeline
- **Adaptive Cruise Control (ACC)** with time-gap based following distance
- **Lane Keeping Assist (LKA)** with proportional steering control
- **Multi-object tracking** with persistent IDs and data association
- **Safety monitor** with multi-layer constraint enforcement

#### Production Infrastructure
- **Comprehensive error handling** with domain-specific exceptions
- **Structured logging** with performance metrics and safety event tracking
- **Configuration validation** with type checking and range validation
- **Metrics collection** for observability and monitoring
- **Docker support** with multi-stage builds and health checks
- **CI/CD pipeline** with GitHub Actions
- **Comprehensive test suite** with 32+ unit tests

#### New Modules
- `src/adas/exceptions.py` - Exception hierarchy for error handling
- `src/adas/validation.py` - Input validation utilities
- `src/adas/logger.py` - Structured logging framework
- `src/adas/safety.py` - Safety monitor and constraint enforcement
- `src/adas/metrics.py` - Performance metrics collection
- `src/adas/models.py` - Domain models (renamed from types.py)

#### Documentation
- `README.md` - Complete project documentation
- `ARCHITECTURE.md` - System architecture and design documentation
- `DEPLOYMENT.md` - Production deployment guide
- `CHANGELOG.md` - This file
- `config.example.json` - Configuration template
- `Dockerfile` - Production container image
- `docker-compose.yml` - Container orchestration
- `Makefile` - Development and build automation

### 🔄 Changed

#### Breaking Changes
- **Renamed `types.py` to `models.py`** to avoid Python stdlib conflicts
- **Controller is now stateless** - current speed passed as parameter
- **Pipeline.step()** now requires `current_speed_mps` parameter
- **Enhanced configuration** with more granular control parameters

#### Improvements
- **Tracking**: Optimized with `math.hypot()`, added error handling
- **Planning**: Added time-gap following, improved lane centering
- **Control**: Fixed state management, added deadband, made thread-safe
- **Safety**: Added comprehensive bounds checking and command sanitization
- **All components**: Added validation, logging, and error handling

### 🐛 Fixed

- **Critical**: Fixed circular import caused by `types.py` module name
- **Critical**: Fixed controller state management (was shared across calls)
- Division by zero protection in tracking distance estimation
- Missing error handling throughout the codebase
- No logging in production code (used print statements)
- Configuration loaded without validation
- No safety limits enforcement

### 📊 Metrics

**Code Statistics:**
- 17 Python modules
- 1,751 lines of production code
- 32 unit tests (100% passing)
- 6 test modules

**Performance:**
- <1ms latency per frame (mock implementations)
- 1000+ FPS throughput
- <100MB memory footprint

### 🔒 Security & Safety

- Input validation on all external data
- Bounds checking on all control outputs
- Safety monitor with configurable limits
- Command sanitization before actuation
- Graceful degradation on sensor failures

### 📦 Dependencies

**Runtime:**
- Python 3.10+
- No external dependencies (core functionality)

**Development:**
- pytest (testing)
- ruff (linting/formatting)

### 🚧 Migration Guide

For users of the previous version:

1. **Update imports:**
   ```python
   # Old
   from adas.types import ...
   
   # New
   from adas.models import ...
   ```

2. **Update pipeline calls:**
   ```python
   # Old
   plan, cmd = pipeline.step(frame)
   
   # New
   plan, cmd = pipeline.step(frame, current_speed_mps=speed)
   ```

3. **Update configuration:**
   - Copy `config.example.json` to `config.json`
   - Adjust parameters as needed
   - All configs are now validated

### 🎯 Next Steps

**Planned for v0.2.0:**
- TensorRT integration for GPU acceleration
- ROS/ROS2 compatibility layer
- Data recording and replay tools
- Sensor fusion (radar, lidar)
- Advanced path planning
- Model predictive control (MPC)

### 📝 Notes

This is a reference implementation for development and testing. For production automotive deployment, additional safety certifications (ISO 26262), redundancy, and extensive validation are required.

---

**Full Changelog**: https://github.com/jakhon37/ADAS/commits/codex/check-project
