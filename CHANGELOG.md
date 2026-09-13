# Changelog

## [Unreleased] — 2026-09-13 safety-hardening pass

Four adversarial reviewers re-verified the 0.3.0 claims on the target Jetson from
scratch and proved a set of defects with numerical repro scripts; three of the four
returned "not ready". This entry records what was fixed, what was **demoted**, and
what is still broken. It is deliberately not a release: the version in
`pyproject.toml` is still `0.3.0`.

Re-measured on the Xavier NX on **2026-09-13, 16:47 KST**, GPU mutex held, board
shared with other work (loadavg 2.9): 200 frames of `example.mp4` through YOLOX-Nano
+ UFLD-v2 + tracker + planner + controller + arbiter at **16.0 FPS end to end**
(61.9 ms/frame), 0 perception failures, 136 detections, 154 tracks, lane detected on
100% of frames. Suite: **859 passed, 1 failed** (was 743 passed at 0.3.0; the
failure is real and is listed under *Still broken*).

### Safety — the arbiter stopped doing four unsafe things

- **It no longer disengages itself during normal driving.** `_decide_state`
  incremented the DISENGAGE latch counter on *any* finding, and "findings" included
  ordinary successfully-mitigated clamps — a frame arriving faster than `min_dt_s`,
  a lateral-accel clamp, a steering-rate clamp, a measured jerk. On an empty road at
  20 Hz the arbiter latched the terminal DISENGAGE state at frame 47. Findings are
  now split into three disjoint lists carried through `arbitrate`: `faults` (health:
  perception dropout, missing/implausible ego, non-finite command, missing/non-finite
  plan, invalid or stale timestamp, dt above `max_dt_s`) which alone increments the
  latch; `mitigated` (clamps that worked, and measurements) which forces LIMITED and
  can never reach DISENGAGE; and `hazards` (traffic) which forces LIMITED/MRM.
  `result.violations` is still the concatenation, so nothing stopped being logged.
- **It no longer reduces an emergency brake.** `_synthesise_command` applied a
  5.0/s brake apply-rate limit against its own previous output, so an input brake of
  1.00 came out 0.250 / 0.500 / 0.750 / 1.000 over four frames — 200 ms, about 3 m
  at 15 m/s — while the docstring claimed "brake only ever increased" and the unit
  test encoded the reduction as *expected*. `ArbiterLimits.brake_apply_rate_per_s` is
  removed and the invariant is now structural: a final
  `brake = max(brake, clamp(cmd_in.brake))`. The brake *release*-rate floor is kept;
  it only ever holds the brake on longer. Brake application jerk belongs to the
  controller, which owns the emergency exemption.
- **A bad lane centre can no longer hide a real lead.** `_in_path` short-circuited on
  `TrackedObject.in_ego_lane` — computed by the tracker from the same lane model the
  planner reads — and otherwise tested an image band centred on the lane model with
  no confidence or `is_mock` check. The corridor is now anchored on the **image
  centre** with half width at least `min_in_path_half_width_frac` (0.30 of frame
  width), membership is by box overlap, and a lane model may only *widen* it, never
  move or narrow it, and only when it is not mock, is finite, is inside the frame and
  has confidence ≥ `lane_trust_confidence` (0.50).
- **The AEB backstop is no longer blind on a new track.** The closing-rate estimate
  started at 0.0 and took ~0.55 s to converge, so `lead_speed = ego_speed` was assumed
  exactly when a hazard appeared, collapsing the RSS minimum gap. `_AlphaBetaRange`
  now seeds every (re-)initialisation from `-max(0, ego_speed)` — the safe prior that
  an unknown object is stationary in the world, bounded by ego speed and taken from
  the ego state, not from `TrackedObject.velocity_mps`. Switching range channel now
  re-seeds instead of differencing across the signal discontinuity, which had been
  manufacturing −45 m/s of "closing" for a stationary car.
- **The minimum-risk manoeuvre no longer straightens the wheel in a bend.** In
  MRM/DISENGAGE the steering was replaced by `_last_good_steering`, which was only
  written in NOMINAL — so an MRM entered in a curve commanded steering 0.0 while
  braking, bypassed the lateral-accel ceiling, and regenerated a `steering_rate`
  fault against its own hold every frame. It now holds the last commanded angle,
  re-checks it against the lateral-accel ceiling at the current speed, slews it, and
  straightens only below `mrm_straighten_speed_mps` (1.0 m/s).
- **`_fuse_range` no longer takes `min(pinhole, depth)`.** Four explicit cases: no
  usable second channel → pinhole, no finding at all if the channel is simply off;
  agreement → confidence-weighted blend; disagreement with the second channel
  **nearer** → the pinhole is used until `range_corroboration_frames` (3) consecutive
  disagreeing frames on that track; disagreement **farther** → never adopted. A
  second-channel confidence below `min_range_confidence` (0.35) is discarded outright
  rather than down-weighted. A single-frame phantom close reading at highway speed
  used to produce a full-authority emergency stop.
- **An uncalibrated camera can raise a violation.** `SafetyContext.camera_calibrated`
  and `ArbiterLimits.allow_uncalibrated_range` were added; a `False` camera appends
  `camera_uncalibrated`, floors the state at LIMITED and cuts throttle. **This is not
  wired**: `src/adas/runtime/pipeline.py` does not pass `camera.calibrated`, so it
  does not fire in a live run. See *Still broken*.
- **`SafetyLimits.to_arbiter_limits` stopped dropping limits on the floor.** It
  silently discarded 14 fields, two of which can latch the terminal DISENGAGE state.
  All are now projected, and a test enumerates `dataclasses.fields(ArbiterLimits)`
  and fails on any field with no `SafetyLimits` counterpart.

### The frame loop and the runner

- **Leaving the loop is a command.** Every loop exit is now classified
  (`completed` / `stopped` / `eof` / `source_lost` / `pipeline_dead`) and
  `PipelineRunner._settle` runs on all of them. A clean exit emits one
  zero-throttle release; an **unsafe** exit re-emits the arbiter's minimum-risk
  command once per nominal period for `failsafe_hold_s` (default 1.0 s) so the brake
  ramp actually completes. Previously a mid-stream sensor loss took `break` and left
  the last throttle latched on the actuators for ever — the exact failure the runner
  docstring claimed to have fixed. The hold is **bounded** and hands back to the
  caller; `deploy/adas.service` is the supervisor.
- **Tracks coast honestly through a perception dropout.** `ADASPipeline._coast_tracks`
  runs a predict-only tracker step: no detection is fabricated and no track is
  corrected, but covariance grows, `time_since_update` rises, the range estimate
  becomes `RangeSource.UNAVAILABLE` and tracks are deleted at `max_missed`. The
  previous build froze every track at its pre-dropout position with
  `time_since_update == 0`, so the recovery frame associated a real measurement
  against an N-frame-stale prediction claiming full confidence.
- `failsafe_command` now stamps `timestamp_s`, so a run of failed frames no longer
  leaves the recovery frame flagged `timing_stale_input` with its `dt` forced to
  `max_dt_s`.

### Planning and control

- **The planner's AEB decision can now be executed.** `LongitudinalPlanner.plan`
  rate-limited its *own* AEB target at `emergency_decel_mps2 · dt` (0.4 m/s per 50 ms
  frame), so the published target trailed the vehicle, the controller's residual
  speed error stayed ~0.4 m/s, and it commanded essentially no brake — the arbiter
  was the only thing in the system that actually braked. The AEB branch now publishes
  0 m/s on the frame it fires, with no downward rate limit; the comfort branch keeps
  its `max_decel_mps2` limit and both branches keep the upward `max_accel_mps2` limit.
- **The controller gained an emergency feed-forward.** Between the PI law and the
  jerk limit, an emergency frame demands the deceleration that erases the whole
  remaining speed error inside `emergency_stop_time_s` (1.0 s), saturated at
  `brake_authority_mps2`. The comfort deadband is skipped and throttle is forced to 0
  on every emergency frame. Measured on the reviewers' cut-in repro, the controller's
  own brake went from 0.094 to 1.000 within four frames.
- **Log flood fixed in the planner layers.** A `LogGate` (one line on entry, at most
  one repeat per period carrying the suppressed count, one line on exit) replaced the
  per-frame WARNING for the permanent `ego.source: none` state. Measured: 1200
  WARNING lines over 1200 frames → 1. `src/adas/control/arbiter.py` has **not** been
  converted; see *Still broken*.

### Perception

- **MiDaS is demoted, not repaired.** It was published as a metric range channel —
  `RangeEstimate.distance_m` in metres, `source=DEPTH_MODEL`, confidence 0.69–0.72 —
  and the arbiter substituted it into the range that TTC, RSS and AEB are computed
  from. Measured per-object rank correlation against the reference range: **0.14–0.25
  across eight sampling strategies plus a 3×-zoomed second inference pass**, a true
  5.3–61.5 m spread compressed to 3.7–13.4 m, and at frame 300 the 54 m car reported
  nearer than the 13 m car. `DepthRangeChannel.update()` now returns
  `RangeSource.UNAVAILABLE` with confidence 0 for every box.
  `publish_metric=True` is a *request*: a rolling self-audit computes Spearman over
  the last 240 (reference range, sampled disparity) pairs and metres flow only while
  that is ≥ 0.80 over ≥ 24 pairs. It fails closed. A new `OrdinalDepth` record
  (disparity, rank, of, normalized) is exposed instead — no `distance_m`, no
  confidence, no `source`, and not a `RangeEstimate`, so the arbiter's range fusion
  cannot consume it by accident. The road-plane affine fit still runs and is still
  reported in `stats()` (relative residual 0.041–0.076); it is the per-object
  sampling that carries no range information.

### Operations and provenance

- **Engines are verified at load.** `models/MANIFEST.json` was documentation only:
  no production caller passed `expected_sha256`, so a swapped or corrupted `.engine`
  loaded silently into the detector that feeds AEB. `trt_engine.verify_engine_file`
  now checks the digest and the manifest/runtime TensorRT version *before*
  deserialisation, memoises per (path, size, mtime), and records every verified
  engine. `TrtEngine.__init__` defaults `verify_manifest=True`, so all call sites are
  covered. There is no environment-variable bypass. An engine the manifest does not
  list logs a WARNING and loads as unverifiable.
- **`/healthz` reports real memory.** `HealthState.refresh_process()` had zero call
  sites, so `ram_mb` was always `null` and `adas_ram_mb` scraped as the literal `0`
  for a process holding over a gigabyte. It is now refreshed from the frame loop and
  from every probe and scrape, rate-limited to 1 s (measured cost 0.3 ms). An
  unmeasurable RAM exports `NaN`, never `0`. `/healthz` also gained an
  `engine_sha256` object.
- **Event log schema 1 → 2.** `t` is stamped unconditionally from `time.monotonic()`
  inside the writer, so one file no longer mixes two clock domains from two callers;
  `seq` continues from the last record already in the file at startup, so it is the
  authoritative order across a restart; a new 12-hex-digit `run` field distinguishes
  two writer instances. The claim that `boot` "makes `t` orderable across restarts"
  was false — the kernel monotonic clock restarts at zero at boot — and is removed.
- `pyproject.toml` no longer sets `addopts = "-q"`. Combined with the documented
  `pytest -q`, pytest read `-qq` and printed no pass/fail summary at all.
- The engine-missing message no longer prints the same directory twice; it names the
  absolute file paths actually tried.

### Documentation honesty

Every headline number in `README.md`, `QUICKSTART.md`, `DEPLOYMENT.md`,
`docs/JETSON.md` and `models/README.md` was re-measured on 2026-09-13 and replaced;
where a claim could not be reproduced it was deleted or explicitly marked
unverified. Specifically corrected:

- "743/743 tests passing" → 859 passed, 1 failed, with the failure named.
- "15.3 FPS" → 16.0 FPS, with the backend matrix and the loadavg it was taken at.
- The MiDaS row's "Real, off by default … affine-aligned per frame against
  road-plane anchors. A cross-check, not a metric sensor" → demoted, with the
  measured correlation.
- "the arbiter's rate-shaped fail-safe" → the braking direction is not rate-shaped
  downward at all any more.
- "in-path test: `in_ego_lane`, else its own wider image band" → image-centre anchor,
  lane as a widen-only second anchor.
- "the nearer of (blend, pinhole, depth) is used" → the four-case rule.
- `models/MANIFEST.json`'s `yolox_nano.preprocessing.verified_empirically` claimed
  "BGR + raw 0..255 gave 10 detections above 0.30" on frame 150; re-running the
  shipped detector at that threshold yields **3** after NMS. The max-confidence
  figure (0.640) reproduces exactly, so the preprocessing conclusion stands — the
  count was a pre-NMS number and the manifest did not say so.

### Still broken, and deliberately left visible

- `tests/test_integration.py::test_record_and_replay_round_trip` **fails**.
  `RecordedDetector.infer` reads `getattr(frame, "frame_id")` but `ADASPipeline.step`
  hands it `frame.rgb`, so every replayed frame returns zero detections and the replay
  re-runs the decision layers on an empty road. The test passed before only because
  the recorded side also happened to command brake 0. The replay tooling is not a
  usable regression harness until `src/adas/tools/replayer.py` is fixed.
- The `camera_uncalibrated` violation is implemented and tested but **not wired**:
  `src/adas/runtime/pipeline.py` does not pass `camera.calibrated` into the
  `SafetyContext`.
- The arbiter still logs one WARNING per frame in a steady degraded state.
  `LogGate` exists and is importable; `arbiter.py` has not been converted.
- The new `SafetyLimits` fields (including every threshold that can latch DISENGAGE)
  are reachable programmatically but **not from a YAML config file**:
  `adas.cli.build_safety_limits` and `SafetyConfig` have no keys for them.
- `failsafe_hold_s`, `emergency_stop_time_s` and the log-gate periods are code-level
  fields with no CLI flag or config key.
- The arbiter's in-path corridor is an **image-space** band, not a metric ego-width
  corridor projected through a calibrated camera. It does not narrow with range.
- The arbiter's own range filter is still not advanced through a perception dropout:
  `_assess_lead` sits behind the `if perception.ok` gate, so a long dropout can still
  read as a range jump on the recovery frame.
- Metric depth is not restored and is not restorable with this model at this
  resolution on this footage. It needs a calibrated camera **and** a better model.
- No camera, no soak test, no systemd start, no certification. See
  *What is not production ready* in `README.md`.

---

## [0.3.0] - 2026-09-13

> The 0.3.0 numbers below are the historical record of that release and were correct
> when it was cut. They have since been superseded — see the hardening pass above for
> the 2026-09-13 re-measurement (16.0 FPS, 859 passed / 1 failed).

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
