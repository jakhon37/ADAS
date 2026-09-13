# Longitudinal safety specification

This document is the specification that `tests/scenarios/` encodes and
`tests/test_scenarios.py` enforces. It exists because the longitudinal safety
path has been rewritten three times without one, and each rewrite was tuned
against whichever ad-hoc script the previous reviewer happened to write. The
result oscillated: the build was advisory, round 1 was authoritative and braked
for nothing, round 2 removed the phantom and stopped braking for things that
were there.

The specification is deliberately written as physics and requirements, not as a
description of what the code does. **No expectation in the harness was derived
by running the arbiter and writing down its output.** If a scenario fails, the
harness is telling you something; the fix is in `src/adas/control/`, not in
`tests/scenarios/library.py`.

---

## 1. Scope

**In scope.** The longitudinal decision path as actuated:

```
BehaviorPlanner -> PIDLikeLongitudinalController -> SafetyMonitor/SafetyArbiter -> actuators
```

with the **arbiter's** command being what reaches the actuators, and a minimal
lateral path (a steering command, a lane-centre measurement, a bend) only where
it is needed to expose a coupling between the two.

**Out of scope.** Perception itself. The harness replaces the detector, the
tracker and the lane estimator with a simulated sensor, because the question
"does the safety path do the right thing given what it was told?" cannot be
answered while the input is also unknown. Perception *characteristics* — range
noise, detection misses, dropouts, re-identifications, lane error — are in
scope, and are declared per scenario.

Nothing here needs a GPU, a TensorRT engine, a camera or a recording.

---

## 2. The reference vehicle

Every judgement in this document rests on these numbers. They are stated as
claims about a vehicle and a road, and any of them can be argued with — but
only explicitly, by changing `tests/scenarios/plant.py` and bumping
`report.SPEC_VERSION`.

| Quantity | Value | Where the number comes from |
|---|---|---|
| Timestep | 0.05 s (20 Hz) | The pipeline's nominal frame period. Fixed; the harness never reads a clock. |
| Max braking authority | 8.0 m/s² | Tyre–road friction on dry asphalt, `mu*g` with `mu ≈ 0.82`. |
| Max acceleration | 2.5 m/s² | Mid-size passenger car at part throttle, 0–100 km/h in ~11 s. |
| Brake rise time | 0.15 s to 90% | Hydraulic build-up plus pad bite. Euro NCAP AEB protocols and ISO 15622 use 0.1–0.2 s. Modelled as a first-order lag. |
| Throttle rise time | 0.35 s to 90% | Powertrain torque response, slower than the brake. |
| Wheelbase | 2.8 m | Mid-size passenger car. |
| Full-scale road wheel | 0.436 rad (25°) | The controller normalises steering against 25°. |
| Steering rise time | 0.10 s to 90% | Steering actuator. |
| Lane width | 3.5 m | Motorway lane; half-width 1.75 m. |

No grade, no drag, no rolling resistance: over the 5–20 s a longitudinal
scenario lasts, drag on a passenger car is under 0.4 m/s² and grade is zero by
assumption. Omitting them keeps the oracle's kinematics closed-form and exact.

Derived stopping distances at full authority, quoted throughout the library:

| Speed | `v²/2a` | plus brake rise | plus 2.0 m clearance |
|---|---|---|---|
| 10 m/s | 6.25 m | 6.9 m | 8.9 m |
| 15 m/s | 14.1 m | 15.0 m | 17.0 m |
| 20 m/s | 25.0 m | 26.3 m | 28.3 m |
| 25 m/s | 39.1 m | 40.7 m | 42.7 m |

---

## 3. Margin conventions

These are the only places a number is allowed to be a judgement call, so they
are all in one list.

| Name | Value | Meaning and justification |
|---|---|---|
| `CONTACT_GAP_M` | 0.0 m | The gap is bumper to bumper, so zero is contact. **Avoidability is judged against contact**, not against comfort: a collision the vehicle could physically have avoided is a failure even if avoiding it would have been unpleasant. |
| `REQUIRED_CLEARANCE_M` | 2.0 m | The clearance a *correct* intervention preserves. `required_decel` is the deceleration that stops with this much room left. 2.0 m is the standstill clearance a driver leaves, and the smallest gap at which a camera range estimate is still meaningful. |
| `COMFORT_DECEL_MPS2` | 3.0 m/s² | Above this a passenger notices. An ordinary headway law would not go here. |
| `EMERGENCY_DECEL_MPS2` | 3.5 m/s² | **The boundary.** At or above this, a command is an emergency intervention and needs kinematic justification. Half a metre above comfort so a comfort-limited ramp cannot be mistaken for an AEB by rounding. |
| `NEGLIGIBLE_DECEL_MPS2` | 1.0 m/s² | Below this true requirement there is no hazard. Emergency authority applied while the requirement is below this floor is a **phantom**. |
| `JUSTIFICATION_MARGIN_FACTOR` | 1.5 | A demand up to 1.5× the theoretical minimum is competent engineering: a 20% range under-estimate is 25% extra demand (the requirement goes as `v²/2d`), and the 0.15 s brake rise and a frame of latency account for the rest. |
| `JUSTIFICATION_TOLERANCE_MPS2` | 0.5 m/s² | Additive slack on top of the factor; roughly the discretisation of a jerk-limited brake. |
| `JUSTIFICATION_WINDOW_FRAMES` | 10 (0.5 s) | A brake that works destroys the evidence for itself. Judgements look back half a second — the timescale of a jerk-limited release — so the tail of a successful stop is not scored as a phantom. |
| `STANDSTILL_MPS` | 0.5 m/s | Below this the ego is stopped, and holding the brake is not an intervention. |

Two assumptions about the lead appear, and conflating them is how a harness ends
up demanding clairvoyance:

* **Causal** (`required_decel`, and therefore justification and phantom
  detection): the lead holds its *current* acceleration until it stops. That is
  measurable this frame.
* **True future** (`last_avoidance_frame`, and therefore lateness): the lead's
  actual script. Used only to bound how *late* an intervention was, never to
  require an earlier one.

---

## 4. Requirements

### R1 — No collision when one was avoidable

If, at the first frame on which the lead was within sensor range, full-authority
braking would have avoided contact, then contact is a failure.

Diagnosis: `collided`, with the last frame from which full braking still worked.

### R2 — No late intervention

If the oracle says the true requirement reached `COMFORT_DECEL_MPS2`, the
system's first emergency-grade command must arrive no later than
`last_avoidance_frame`.

Diagnosis: `late_intervention` (by N frames / N seconds), `missed_intervention`.

### R3 — No intervention when none is warranted

The system must not command `EMERGENCY_DECEL_MPS2` or more, and must not enter
`MIN_RISK_MANEUVER`, on a frame where perception is healthy, the ego is moving,
and the true requirement has been below `NEGLIGIBLE_DECEL_MPS2` for the whole
justification window — **unless** the frame belongs to an unbroken run of
braking that began when the requirement was real.

This requirement carries exactly the same weight as R1. Round 1 satisfied R1 and
failed R3; round 2 the reverse. A harness that checks only one is how that
happened.

Diagnosis: `phantom_intervention`, with the frame, the state, the commanded
deceleration, the true gap, the true closing rate and the true requirement.

### R4 — Proportionate intervention

While an intervention *is* warranted, the commanded deceleration must not exceed
`max(EMERGENCY_DECEL, need × 1.5 + 0.5)` where `need` is the worst true
requirement in the justification window.

Over-braking is not a free safety margin. A follower keeping its own 2 s gap and
taking 1 s to react can absorb a 5 m/s² lead deceleration and cannot absorb
8 m/s²: braking harder than the situation requires transfers the collision to
the vehicle behind.

Diagnosis: `disproportionate_brake`. A scenario may allow a bounded number of
frames (`unjustified_brake_frames_allowed`) where a transient is genuinely
defensible — for example the first frames of a brand-new track, which carry no
rate history. That allowance never excuses a phantom.

### R5 — Recovery

Once every disturbance is over — the kinematic hazard has cleared, perception is
healthy again, and no present lead is going undetected — the system must return
to `NOMINAL` within the scenario's bound and **stay there**.

"Stay there" is load-bearing: a system oscillating in and out of a degraded
state has not recovered.

All three of the self-sustaining feedback loops in this module's history
(`steering_rate -> plan_accel -> lane_departure`) are failures of this
requirement and of no other.

Diagnosis: `failed_to_recover`, `slow_recovery` (by N frames / N seconds), with
the violations still latched on the last frame.

### R6 — Bounded authority per situation

A scenario may cap the safety state and the commanded deceleration. Two caps
recur:

* On an **empty road with valid ego state and healthy perception**, the state is
  `NOMINAL` and the brake is zero. There is no measurement that differs from a
  clear road, so any degradation is generated inside the system.
* A **minimum-risk manoeuvre is a controlled stop at ~3.5 m/s²**, not an
  emergency stop. Nothing was seen, so nothing warrants full authority.

Diagnosis: `over_braked`, `over_authority_state`.

### R7 — DISENGAGE is not a minimum-risk state for a moving vehicle

`DISENGAGE` hands the vehicle back to nobody. There is no driver in this loop.
While the ego is moving, a loss of perception — however prolonged — requires a
controlled stop, not a hand-back.

Diagnosis: `disengaged`.

### R8 — Speed is not to be given away

Braking that no measurement justifies is measurable in metres per second, not
only in state labels. Scenarios where nothing is closing carry a floor on the
true ego speed. Round 1's closed-loop failure was exactly this: 20 m/s to 0
behind a car that never moved, while every state label looked defensible frame
by frame.

Diagnosis: `speed_loss`, `final_speed`.

### R9 — In-lane behaviour, including under a minimum-risk manoeuvre

The ego must stay within 1.75 m of the lane centre (half a 3.5 m lane), on a
bend and while decelerating to a stop. A vehicle that holds the wheel straight
while the road bends deviates by `s²/2R`.

Diagnosis: `lane_departure`.

### R10 — Leaving the loop is not a command

Whatever was last written to the actuators stays latched there. A clean exit, a
bounded frame count, a `SIGTERM` and a source loss are all the same event from
the actuators' point of view. At exit the system must **write** something; that
something must have zero throttle; and it must not release a brake the previous
frame was applying.

Diagnosis: `no_exit_command`, `exit_throttle`, `exit_released_brake`,
`exit_no_brake`.

### R11 — A lane estimate may widen the hazard corridor, never move or narrow it

More lane means more places a threat can be. A lane detector that fails towards
the kerb must not be able to hide a car that is directly in front of the bumper.
Enforced indirectly, through R1 and R2 in a scenario where the lane estimate is
mock and wrong by 460 px while a real obstacle closes.

### R12 — A change of measurement is not a change of the world

A track-id change carries no information about motion. A 10 m step in a reported
range in one 50 ms frame implies 200 m/s of closure, which no road vehicle
produces. Neither may command an intervention.

---

## 5. How the harness works

```
WorldState (truth) -> Sensor -> Observation -> planner -> controller -> arbiter
       ^                                                                   |
       +-------------------- actuated command ---- Plant <-----------------+
```

* `plant.py` — the deterministic vehicle and world. Fixed timestep, first-order
  actuator lags, scripted lead, bicycle-model lateral, seeded noise.
* `oracle.py` — pure kinematics over the true state history. Closed-form: the
  minimum gap under a constant deceleration is evaluated at the exact candidate
  times (phase boundaries and the relative-speed zero), so no timestep can hide
  it. Avoidability is answered by re-running the *same plant* with the brake
  pinned at 1.0, so the actuator lag is included.
* `scenario.py` — the declarative scenario, the closed loop, and `evaluate()`.
* `library.py` — the scenarios, each carrying its physics argument as a string
  the report prints.
* `report.py` — the table and the JSON artifact.

Running it:

```bash
# table only
PYTHONPATH=src:. python3 -m tests.scenarios.report

# table plus machine-readable artifact
PYTHONPATH=src:. python3 -m tests.scenarios.report --json build/safety_acceptance.json

# one scenario, with every physics argument printed
PYTHONPATH=src:. python3 -m tests.scenarios.report --only bend_mrm_and_recover --verbose

# under pytest, so a regression fails the suite
PYTHONPATH=src python3 -m pytest tests/test_scenarios.py -p no:warnings
```

The whole library runs in about 30 s on the Jetson Xavier NX, CPU only.

---

## 6. Historical failures and the scenario that now guards each

| Failure | When | Guarding scenario(s) |
|---|---|---|
| Advisory monitor: the pipeline actuated commands the monitor had rejected | build | The harness actuates the **arbiter's** command in every scenario; nothing else can be actuated. |
| Phantom full-authority AEB for a lead at constant range (42/400 MRM frames, 6 frames at brake = 1.00; closed loop 20 m/s → 0 behind a car that never moved) | round 1 | `constant_range_motorway`, `constant_range_close_follow` |
| Phantom AEB on the first frame of a new track (range-rate seeded at the ego speed) | round 1 | `reid_during_steady_follow`, and the frame-0 checks in every steady-follow scenario |
| Phantom AEB after a range re-seed | round 1 | `range_jump_during_steady_follow` |
| DISENGAGE on an empty road with a valid ego state | round 1 | `empty_road_nominal`, `empty_road_dropout`, `source_loss_mid_run` |
| A mock lane centre hides a real lead | round 1 | `lane_error_with_closing_lead` |
| Missed braking against a lead braking at 6 m/s² from 15, 20, 25 and 30 m | round 2 | `lead_brakes_6mps2_from_15m` / `_20m` / `_25m` / `_30m` |
| Self-sustaining loop: `steering_rate -> plan_accel -> lane_departure` | rounds 1–3 | Every scenario with `recover_within_frames`: `lead_brakes_then_leaves_lane`, `cutin_close_at_12m`, `cutin_moderate_at_25m`, `empty_road_dropout`, `bend_mrm_and_recover`, `range_jump_during_steady_follow` |
| Held or zeroed steering during a minimum-risk manoeuvre | round 3 | `bend_nominal`, `bend_mrm_and_recover` |
| ADAS-DEC-21: a latched command after the loop ends | build | `terminate_during_emergency`, `source_loss_mid_run` |

Ordinary cases, which exist so that the guards above cannot be satisfied by
simply never intervening: `stationary_*` (four speeds and ranges),
`lead_brakes_gently_2mps2`, `lead_accelerates_away`, `cutin_moderate_at_25m`,
`dropout_mid_approach`.

---

## 7. Baseline: the current arbiter

Measured against commit `1ce4886` ("Arbiter round 2: phantom AEB removed,
missed-braking regression introduced"), spec version 1.0.

**6 passed, 19 failed, 25 total.**

Passing: `lead_brakes_gently_2mps2`, `lead_accelerates_away`,
`constant_range_motorway`, `empty_road_nominal`, `reid_during_steady_follow`,
`bend_nominal`.

Failures by diagnosis:

| Diagnosis | Scenarios | What it means |
|---|---|---|
| `disproportionate_brake` | 10 | The response is bang-bang. Full authority (8.0 m/s²) is commanded where 4.7–5.0 m/s² is required, from 40 m and 62 m out. |
| `phantom_intervention` | 7 | Emergency authority with no kinematic warrant. Frame 0 of a new in-path track at a constant 15–20 m gap still commands 5.0 m/s²; a minimum-risk manoeuvre persists 9 frames after perception recovers. |
| `slow_recovery` | 5 | Return to NOMINAL takes 4.1 s (`empty_road_dropout`), 6.9 s (`bend_mrm_and_recover`), 12.3 s (`cutin_moderate_at_25m`) after the last disturbance. |
| `over_braked` | 2 | 5.0 and 8.0 m/s² where the scenario permits 3.5. |
| `over_authority_state` | 2 | DISENGAGE reached where MIN_RISK_MANEUVER is the ceiling. |
| `failed_to_recover` | 1 | `cutin_close_at_12m` never returns to NOMINAL; it ends LIMITED with `plan_accel_8.56_above_3.00_arbiter_induced` still latched — the self-sustaining loop, caught. |
| `disengaged` | 1 | `source_loss_mid_run` latches DISENGAGE at frame 99, 2 s into a permanent sensor loss, while still moving. |

**No collisions and no missed interventions.** In this harness's closed loop the
current arbiter stops in every case where stopping is possible, with a minimum
true gap of 2.25 m in the worst case (`lead_brakes_6mps2_from_15m`). That does
*not* contradict the round-2 report of collisions from 15, 20, 25 and 30 m: this
harness drives the full planner → controller → arbiter chain with the planner's
lane gate configured at 0.25 and the arbiter's at 0.30, and it actuates the
arbiter's command. A different wiring — in particular the planner's shipped
default of `ego_lane_half_width_frac = 0.0`, or actuating the controller's
command instead of the arbiter's — is a different system and may well collide.
That the two disagree is itself an argument for having one specification instead
of a folder of scripts.

The failure profile is coherent: **the current arbiter is not blind, it is
indiscriminate.** It intervenes when it should, it also intervenes when it
should not, it intervenes harder than the kinematics justify, and it is slow to
let go.

---

## 8. Changing this specification

The harness fails loudly and that is the point. Before changing any expectation:

1. Change **this document** first, with the physics argument for the new number.
2. Change the scenario's `physics` string to match. The report prints it next to
   every failure; a justification that no longer matches the expectation is
   worse than none.
3. Bump `report.SPEC_VERSION` if a margin convention, an authority limit or a
   plant constant changed. JSON artifacts from different spec versions are not
   comparable.

Do **not**:

* relax an expectation because the current code fails it;
* add a scenario whose expected outcome you determined by running the arbiter;
* delete a scenario that guards a historical failure. Add to the table in
  section 6 instead — that table is the record of what this codebase has already
  got wrong once.
