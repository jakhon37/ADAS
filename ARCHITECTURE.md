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
* **`depth.py`** — MiDaS as a *relative* depth channel, **demoted from a metric
  range channel** and off by default. The road-plane affine fit still runs (its
  anchors genuinely read no detection box, and its relative residual is 0.041–0.076),
  but the per-object output is sampled from pixels *inside* a box the detector drew,
  and that sample was measured to carry almost no range information: Spearman 0.14–0.25
  against the reference range over the replay clip, against the 0.80 bar the channel
  now holds itself to. `update()` therefore returns `RangeSource.UNAVAILABLE` with
  confidence 0 for every box. `publish_metric=True` is a *request*: a rolling audit
  over the last 240 (reference range, disparity) pairs must measure ≥ 0.80 Spearman
  over ≥ 24 pairs, and it fails closed. A separate `OrdinalDepth` record — no
  `distance_m`, no confidence, no `source`, not a `RangeEstimate` — carries the
  ordinal signal so the arbiter's range fusion cannot consume it by accident.
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

`LongitudinalPlanner` runs **two laws and takes the larger deceleration**, and
the split is the point:

* a **constant time gap** for following (`d_desired = d₀ + T·v`), whose
  contribution is capped at `headway_decel_mps2`. Headway keeping is not
  collision avoidance and must not be able to produce an emergency;
* an **evidence-gated avoidance law**, `evidence.required_decel_mps2`, gated on
  the four-sigma lower confidence bound of a closure the planner measures itself
  and sized on the unbiased estimate of the same closure.

It publishes two numbers and they are not interchangeable. `target_speed_mps` is
a comfort request the throttle may serve; `decel_demand_mps2` is the
authoritative braking figure, already jerk shaped. The published target passes
through one rate limiter (`_rate_limited_target_mps`): rising is always bounded
by `max_accel_mps2·dt` because the vehicle cannot follow a step and the only
thing a step does is wind up the controller's integrator, falling by
`max_decel_mps2·dt`, and an **AEB frame is exempt and publishes 0 m/s at once** —
a target that trailed the vehicle down during an emergency is how the primary
path came to contribute 0.06 m/s² while the arbiter did all the braking.

An invalid ego speed degrades to hold-and-ramp-down and reports
`ego_speed_unavailable` — it never guesses. A perception dropout holds the last
avoidance demand for `blind_hold_frames` and then makes a minimum-risk stop at
the comfort rate; a detection miss holds it for `miss_hold_frames`. Neither a
dropped frame nor a missed detection is evidence of an empty road.

**The primary path can stop the vehicle on its own.** That is a structural claim,
not a tuning one, and the harness holds it to it:
`primary_alone_stops_for_stationary` discards the arbiter's command entirely and
requires the planner and controller to stop for a parked car — measured
clearance **2.53 m**. When the only brake in a system is its safety monitor,
every tuning change has to trade phantom braking against missed braking, because
there is nothing else to carry the ordinary case; that is the oscillation this
module was redesigned out of.

`LateralPlanner` has two laws behind one interface. The non-metric law
(shipping) is speed-scheduled proportional control on the normalised pixel error
and is explicitly labelled `is_metric=False` with `lateral_error_m=None`. The
metric Stanley law needs a calibrated camera and
`CameraGeometry.from_camera_config` refuses an uncalibrated one, because an
assumed mount height would turn `lateral_error_m` into a fabricated measurement
wearing metric units. Both are capped by `δ_max = atan(a_lat_max·L/v²)`.

### `adas.control.evidence` — the arithmetic both layers share

New, and the reason the two layers can be co-designed without being coupled.
It holds the range estimation (`RangeEvidence`, `EvidenceBook`), the stopping
kinematics (`required_decel_mps2`, `stopping_distance_m`), the jerk shaper and
the sub-emergency guard, as pure functions and small stateful objects with no
knowledge of either consumer.

`LongitudinalPlanner` and `SafetyArbiter` each construct their **own**
`EvidenceBook`, fed from their **own** lead selection. Shared maths, disjoint
state, disjoint decisions: a corrupted window in one cannot reach the other, and
`EvidenceLimits` is a separate instance on each side so a deployment can demand
different amounts of evidence of the primary path and of the backstop without
either setting reaching the other.

### `adas.control` — controller and arbiter

`PIDLikeLongitudinalController` is stateful: deadband with hysteresis on actuator
selection → PI producing an **acceleration** (gains are 1/s and 1/s²) with
clamping anti-windup → jerk limit → pedal map through the authority constants →
pedal rate limits. It validates its own output.

`SafetyArbiter.arbitrate(plan, command, SafetyContext) -> ArbitrationResult` is
the authority. It is independent of the planner *by construction*:

| quantity | planner | arbiter |
|---|---|---|
| lead selection | nearest in-lane | largest REQUIRED DECELERATION, then nearest, over the RAW track list |
| range rate | its own `EvidenceBook` | its **own** `EvidenceBook`, a separate instance with separate state |
| in-path test | `in_ego_lane`, else lane centre | METRIC corridor from the object's own `lateral_offset_m`, widened by `lateral_gate_margin_m`; **never** reads `in_ego_lane` |
| lane model | the geometry it steers on | a *widen-only* second anchor, and only if not mock, finite, in frame, confidence ≥ 0.50 |
| kinematics | plan over a horizon | differenced from measured ego speed |
| headway rule | time gap | RSS minimum gap |

Both sides share the *arithmetic* in `adas.control.evidence` and share none of
the *state*: each constructs its own `EvidenceBook` from its own lead selection,
so a corrupted window in one cannot reach the other. Three properties of that
module are there because the shipped arbiter got each of them wrong:

* **measurement noise is estimated from THIRD differences of the raw range.** A
  third difference annihilates a quadratic exactly, so a lead holding a constant
  deceleration contributes nothing to the noise estimate. Taking it from fit
  residuals instead means a braking lead reads as a noisy stationary one, loses
  its deceleration credit, and is driven into.
* **the noise estimate is pooled over the whole run**, not taken from the last
  window. A four-sample fit's residuals carry two degrees of freedom and collapse
  to nearly nothing several times in a 300-frame run; every standard error
  computed from them collapses with it and the confidence bound stops bounding.
* **the range window is stamped with the CAPTURE time**, from
  `SafetyContext.measurement_t_s`, not with the decision clock. See the note
  under *Latency* below.

**What is not diverse.** `SafetyContext.tracks` is the tracker's output and
`EgoState` is the planner's ego state, so a detection perception never produced is
invisible to both, and a wrong ego speed fools both identically. The arbiter is a
second opinion on the *decision*, not a second sensor.

Range fusion (`_fuse_range`) is **stateless**, and its rule is two lines
(45 executable lines with the argument written out): when a second
channel is present, finite and above `min_range_confidence` (0.35), the fused
range is the NEARER of it and the pinhole if the two agree within
`range_disagreement_frac` (0.30); otherwise the pinhole is used, the
disagreement is reported and the state is degraded. Two ranges twelve times
apart are not two opinions about one distance — at least one channel is broken,
and the pinhole is the one with a geometric derivation and a calibration behind
it, so the response is to report and degrade rather than to adopt the broken
channel.

This replaced an 87-executable-line state machine carrying five per-track
dictionaries: a
confidence gate with hysteresis, an asymmetric adoption/dwell counter, a
disagreement corroboration streak and a confidence-weighted blend. Two of its
three parts were inert and the third was a latch. The blend returned a value
strictly between the two channels and the very next expression took `min` of the
blend and both channels, which is `min` of the two channels for any weight. The
hysteresis and dwell existed to stop the reported *provenance* flip-flopping (26
times in 400 frames of real footage) — a logging problem, now solved in the
logger. And the corroboration streak delayed adopting a persistently disagreeing
nearer channel by three frames and then adopted it anyway.

The single-frame phantom close reading that motivated the original machinery is
still rejected, and by a stronger mechanism: a 3 m reading against a 40 m pinhole
is a 1233% disagreement, so it never reaches the hazard maths at all, and even if
it did the evidence window would reject it as a range jump beyond
`evidence.jump_m` (3.0 m) and report an unmeasured rate, which authorises no
braking.

`aeb_rate_corroboration_frames` went the same way. It required a measured
emergency to hold for two consecutive frames before it latched a minimum-risk
manoeuvre. It was a latch on evidence and it was redundant: the closure must
already clear four standard errors of zero and the lead deceleration five before
either reaches the hazard classifier. Removing it moved the first minimum-risk
frame from 4 to 3 across the sweep's tightest family with no new phantom, early
or band finding anywhere in the 120-cell gate or the 53-scenario corpus. Both
keys, and the four range-stability keys, are retained so an existing config still
loads, are still range-checked, and are reported inert by
`ArbiterLimits.unused_limits()`.

**Latency.** Every rate here is a slope, and a slope is only as good as its
abscissae. `SafetyContext` carries `measurement_t_s`, the time the measurements
were TRUE, alongside `timestamp_s`; the range window is fitted on the former.
With a constant sense latency the two differ by a constant and a slope does not
care, which is why using the decision clock went unnoticed at the measured 55 ms.
Once latency exceeds a frame period the same capture is republished on
consecutive decision frames, and deduplicating those leaves the survivors stamped
with the decision time of first sight. Measured at 80 ms on a 36 m stationary
approach: a reported closing rate of 9.23 m/s for a true 20 m/s, and a finish
1.17 m from the obstacle against 2.00 m required. `adas.runtime.pipeline` passes
`frame.timestamp_s` for both, so production behaviour was always correct; the
scenario harness now passes `obs.measurement_t_s`, and
`degraded_latency_stationary_20mps_at_36m` keeps it that way.

Findings are split three ways and only one of them can latch the terminal state:

| list | contents | consequence |
|---|---|---|
| `faults` | perception dropout, missing/invalid/implausible ego, non-finite command, missing or non-finite plan, non-monotonic/invalid/stale timestamp, `dt` above `max_dt_s` | increments the DISENGAGE counter |
| `mitigated` | clamps that worked, and measurements: lateral accel, steering rate, plan over-speed/steer/accel, command range clamps, range disagreement/jump/source switch, measured accel and jerk, lane departure, `camera_uncalibrated` | forces LIMITED, can **never** reach DISENGAGE |
| `hazards` | traffic | forces LIMITED / MRM |

`result.violations` is still all three concatenated, so nothing stopped being logged.
Before this split, an ordinary successfully-mitigated clamp incremented the latch and
the arbiter disengaged itself on an empty road at frame 47.

State machine: NOMINAL → LIMITED → MIN_RISK_MANEUVER → DISENGAGE, escalation
immediate, de-escalation only after `recovery_frames` clean frames, DISENGAGE latched
until `reset()`.

**Output shaping.** The output is never more energetic than the input: throttle is
only ever reduced and **brake is only ever increased**, enforced by a final
`max(brake, cmd_in.brake)`. The single exception is a positively EMPTY road —
perception healthy, ego valid, no in-path object now or recently — where the
arbiter may lower an incoming brake to `no_hazard_decel_cap_mps2`, because an
unwarranted 8 m/s² stop on a motorway does not avoid a collision, it manufactures
one behind. A lead merely being *measured* is not that exception and must never
arm it: one of the three redesign candidates attenuated an incoming 1.00 to 0.12
with a benign car 45 m ahead, and the round before it attenuated 1.00 to 0.25 and
drove into the lead. `test_an_incoming_brake_is_passed_through_unattenuated_with_a_lead_in_view`
and its converse pin both directions; the scenario corpus covers only the empty
road, so neither is redundant. There is deliberately no brake *apply*-rate limit here —
that belongs to the controller, which owns the emergency exemption. Only the throttle
apply-rate and the brake *release*-rate floor survive.

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
2. A perception exception sets `PerceptionStatus(ok=False)` and tells the planner
   `perception_valid=False`. No detection is fabricated and no track is corrected,
   but the tracker **is** advanced by a predict-only step (`_coast_tracks`): live
   tracks coast with growing covariance and rising `time_since_update`, their range
   estimate becomes `RangeSource.UNAVAILABLE`, and they are deleted at `max_missed`.
   Freezing them at their pre-dropout position — the previous behaviour — made the
   recovery frame associate a real measurement against an N-frame-stale prediction
   that still claimed full confidence. A detector fault and a lane fault are tracked
   separately. An empty list from a working detector still means the road is clear.
   The arbiter no longer has an alpha-beta filter to advance: its range evidence
   is a window of RAW measurements, and a dropout simply adds no sample to it
   while `blind_hold_frames` holds the last avoidance demand. An extrapolation is
   not a measurement and must never be able to establish the closure that
   authorises an emergency stop.
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

It also classifies every way the loop can *exit* (`completed` / `stopped` / `eof` /
`source_lost` / `pipeline_dead`) and runs `_settle` on all of them. A clean exit emits
exactly one zero-throttle, zero-brake, zero-steering release. An unsafe exit
(`source_lost`, `pipeline_dead`) re-emits the minimum-risk command once per nominal
period for `failsafe_hold_s` (default 1.0 s) so the arbiter's brake ramp completes;
re-emission drives the plant, refreshes health and logs the transition, but
deliberately does **not** tick the watchdog, because the frame id has not advanced and
a runner that lost its source must stay visible to systemd. The hold is bounded and
hands back to the caller — it is not a supervisor. Before this, `break` on a lost
source left the last throttle latched on the actuators exactly as `continue` had.

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
| frame loop exited unsafely | `source_lost` / `pipeline_dead` | runner holds the MRM for `failsafe_hold_s`, then returns |
| engine missing | `PerceptionError` at build | process refuses to start |
| engine digest ≠ `models/MANIFEST.json` | `EngineIntegrityError` before deserialisation | process refuses to start; no bypass |
| depth channel has no ordering skill | audit gate shut | `RangeSource.UNAVAILABLE`, not a low-confidence metre value |

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
