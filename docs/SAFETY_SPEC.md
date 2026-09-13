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

Stopping distances at full authority. The middle column is **measured**, by
driving `plant.Plant` at `brake = 1.0` from each speed until it stops, so it
includes the first-order brake rise rather than estimating it. These four
numbers are the **avoidability boundary for a stationary obstacle**: at 20 m/s a
parked car 25 m ahead cannot be avoided by any system whatever, and one 26 m
ahead can be avoided by exactly 0.15 m. Every scenario in the stationary family
is placed against this column, which is what stops the harness from reporting a
"missed brake" in a situation the arithmetic had already decided.

| Speed | `v²/2a` (ideal) | measured through the plant | plus 2.0 m clearance |
|---|---|---|---|
| 10 m/s | 6.25 m | **6.67 m** | 8.67 m |
| 15 m/s | 14.06 m | **14.70 m** | 16.70 m |
| 20 m/s | 25.00 m | **25.85 m** | 27.85 m |
| 25 m/s | 39.06 m | **40.13 m** | 42.13 m |

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

## 3a. Measured failure boundaries, and how the scenarios are placed on them

The first version of this library was rejected by its own backtest for one
reason above all others: **its scenarios were not on any boundary.** The four
cases that named "round-2 missed braking" as their guard braked the lead at
`start_s = 1.0`, handing the ego a free second at zero closing rate, and all
four PASSED on the commit that collides. The stationary family sat at
25/45/60/75 m while contact occurs below 26 m. A scenario 20 m clear of the
boundary cannot distinguish a system that is 1 m from failing from one that is
20 m from failing, and it cannot see a regression until the regression is
already catastrophic.

Every family is now placed by measurement. The boundaries below were located by
running the sweep and the closed-loop scenario runner in detached worktrees of
the two committed arbiter versions with known failures:

* `25e3ba5` — **phantom braking** (arbiter md5 `d85a8a2b`)
* `1ce4886` — **missed braking** (arbiter md5 `d517e2aa`)

### B1 — Constant-range phantom (`25e3ba5`)

Matched-speed follow, closing rate identically zero, full stack, closed loop.
Emergency authority means commanded ≥ 3.5 m/s² **or** `MIN_RISK_MANEUVER`.

| Ego | Emergency authority fires for | First clean range | Peak commanded at the top of the region |
|---|---|---|---|
| 15 m/s | gap ≤ 26 m | 27 m | 5.88 m/s² at 26 m |
| 20 m/s | gap ≤ 43 m | 44 m | 5.90 m/s² at 43 m |
| 25 m/s | gap ≤ 66 m | 67 m | 5.80 m/s² at 66 m |

Closed loop at 20 m/s the ego is dragged from 20 m/s down to 13.80, 14.83, 15.86
and 17.27 m/s at gaps of 12, 20, 30 and 40 m, and is untouched at 52 and 70 m.
The same measurement on `1ce4886` gives emergency authority only for gaps ≤ 20 m
at 20 m/s, followed by a sustained 3.0–3.3 m/s² all the way out to 45 m — which
is inside the sub-emergency band and was invisible to every assertion in the
first version of this harness.

**Scenarios straddling B1:** `constant_range_12m`, `_20m`, `_30m`, `_40m`
(inside), `_52m`, `_70m` (outside), plus `reid_during_steady_follow` at 40 m and
`range_jump_during_steady_follow`, whose −10 m step lands the apparent range at
42 m, one metre inside the boundary.

### B2 — Lead braking at 6 m/s² from a matched-speed follow (`1ce4886`)

Lead brakes from **frame 0**. Full stack, closed loop, minimum true gap in
metres (negative is contact):

| d₀ | 14 | 16 | 17 | 20 | 26 | 27 | 28 | 30 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| `1ce4886` | +1.01 | **−0.00** | **−0.29** | **−0.65** | **−0.12** | +0.28 | +0.38 | +1.50 | +2.22 |
| `25e3ba5` | +2.79 | +0.34 | +0.31 | +0.40 | +1.56 | +1.82 | +2.12 | +2.59 | +3.11 |

**Avoidable contact for d₀ = 16–26 m at 20 m/s.** The contact is avoidable at
every one of those ranges: both vehicles start matched, so an ego braking flat
out from frame 0 keeps essentially the whole initial gap (16.00 m at d₀ = 16 m).
Neither commit reaches the 2.0 m clearance anywhere below 28 m.

At 15 m/s neither commit collides, and the case is decided on clearance alone:
1.55 m (`1ce4886`) against 3.04 m (`25e3ba5`) at d₀ = 15 m. At 25 m/s **both**
collide for d₀ = 25–40 m.

**Scenarios straddling B2:** `lead_brakes_6mps2_ego20_at_16m`, `_at_20m`,
`_at_26m` (inside), `_at_32m` (outside), `lead_brakes_6mps2_ego15_at_15m`
(clearance-only), `lead_brakes_6mps2_ego25_at_30m` (fails on both commits).

### B3 — Stopped obstacle: the avoidability boundary

The boundary is the measured stop distance from section 2. Minimum true gap:

| Ego, d₀ | best physically available | `25e3ba5` | `1ce4886` |
|---|---|---|---|
| 15 m/s, 14 m | −0.70 (unavoidable) | −0.70 | −1.43 |
| 15 m/s, **15 m** | +0.30 | +0.30 | **−0.43** |
| 20 m/s, 25 m | −0.85 (unavoidable) | −0.85 | −2.21 |
| 20 m/s, **26 m** | +0.15 | +0.15 | **−1.21** |
| 20 m/s, 27 m | +1.15 | +1.15 | **−0.21** |
| 20 m/s, **29 m** | +3.15 | +3.15 | +1.79 (below the 2.0 m required) |
| 25 m/s, 40 m | −0.13 (unavoidable) | −0.13 | −1.83 |
| 25 m/s, **42 m** | +1.87 | +1.87 | +0.17 |

`1ce4886` therefore makes **avoidable** contact at 15 m/s from 15 m, at 20 m/s
from 26–27 m and at 25 m/s from 41 m, and misses the required clearance out to
about 31 m at 20 m/s. Both commits apply full authority to a car 40 m and 75 m
away, where 5.26 and 2.74 m/s² respectively are all that is required.

**Scenarios straddling B3:** `stationary_15mps_at_15m`, `stationary_20mps_at_26m`,
`stationary_25mps_at_42m` (one metre inside the avoidable side of the boundary),
`stationary_20mps_at_29m` (just outside, decided on the 2.0 m clearance),
`stationary_20mps_at_40m` (clear control), `stationary_20mps_at_75m` (the
over-braking control, where the requirement is below comfort).

### Both directions, in every family

Round 1 satisfied only the passive check and round 2 only the aggressive one; a
scenario that can fail only one way is half a test. Every entry in the library
states, in its `physics` string, what a too-aggressive system does to it and
what a too-passive one does. Where a direction is structurally undetectable — at
the avoidability boundary full authority is the only defensible command, so
there is no over-response to find — the entry says so **and names the sibling in
the same family that carries that direction**:

| Family | Passive failure | Aggressive failure |
|---|---|---|
| constant range | `final_speed` / `failed_to_recover`: never letting go | `phantom_intervention`, `over_braked` at the 3.0 m/s² comfort ceiling, `speed_loss` |
| lead braking at 6 m/s² | `collided`, `clearance` | `disproportionate_brake` beyond ten frames — full authority before the lead has shed the speed that warrants it |
| stopped obstacle, boundary cases | `collided`, `clearance` | not detectable; carried by `stationary_20mps_at_75m` |
| stopped obstacle, clear cases | `clearance` | `disproportionate_brake`, `over_braked` above 4.6 m/s² |
| cut-in | `clearance` | `over_braked`, `disproportionate_brake` after five frames |
| dropouts and bends | `failed_to_recover`, `lane_departure` | `disengaged`, `over_authority_state` |

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

### R13 — The sub-emergency band is policed

Deceleration between `COMFORT_DECEL_MPS2` (3.0 m/s²) and
`EMERGENCY_DECEL_MPS2` (3.5 m/s²) is invisible to R3 and R4 by construction: R3
only looks at commands at or above 3.5 m/s², and R4's ceiling is floored at
3.5 m/s². A system can therefore hold 3.4 m/s² for a whole run against a lead
that never moved and satisfy every other requirement in this document. That is
not hypothetical: `1ce4886` holds 3.0–3.3 m/s² on a constant-range follow at
every gap from 25 m to 45 m at 20 m/s.

Where the true closing rate is zero and the headway is safe, no deceleration
above the comfort limit is justified at all, so scenarios in that situation
carry an explicit `max_commanded_decel_mps2 = 3.0`. The sweep carries the same
requirement as an independent third grading, `band_verdict`, reported and gated
alongside the emergency verdict.

Diagnosis: `over_braked` in the scenario suite; `BAND_UNWARRANTED` in the sweep.

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

The whole library runs in about 8 s on the Jetson Xavier NX, CPU only.

### The sweep is the primary gate; this library is the readable specification

The hand-picked library missed **both** of this module's historical failures,
and the envelope sweep found both. That is not an accident of this particular
library: the useful output of a sweep is a *boundary* and the useful output of a
scenario is a *pass*, and a boundary is what tells you whether a change made the
system better or merely moved the failure two metres. So the order of authority
is:

```bash
# THE GATE. 96 cells, ~11 s, non-zero exit on any collision, miss, phantom,
# early or late intervention, or sub-emergency band fault.
PYTHONPATH=src:. python3 scripts/run_safety_sweep.py --gate

# the readable specification, which explains WHY each of those is a fault
PYTHONPATH=src python3 -m pytest tests/test_scenarios.py -p no:warnings
```

`docs/SAFETY_SWEEP.md` documents the sweep. The library exists so that a failing
gate cell can be read as a sentence instead of a coordinate; the gate exists so
that the library cannot be satisfied by a system that happens to be correct at
33 points and wrong everywhere between them.

---

## 6. Historical failures and the scenario that now guards each

Every guard below is placed **inside** the measured failing region of the commit
that exhibits the failure (section 3a), and is paired with a case just outside
it. Guards that were previously outside the region are marked *(moved)* with the
range they moved from.

| Failure | When | Guarding scenario(s) |
|---|---|---|
| Advisory monitor: the pipeline actuated commands the monitor had rejected | build | The harness actuates the **arbiter's** command in every scenario; nothing else can be actuated. |
| Phantom full-authority AEB for a lead at constant range (42/400 MRM frames, 6 frames at brake = 1.00; closed loop 20 m/s → 0 behind a car that never moved) | round 1 | `constant_range_12m`, `_20m`, `_30m`, `_40m` inside the measured 43 m boundary; `_52m`, `_70m` outside it *(moved: the single previous case sat at 52 m, outside the region, and passed on the phantom commit)* |
| Phantom AEB on the first frame of a new track (range-rate seeded at the ego speed) | round 1 | `reid_during_steady_follow` at 40 m *(moved from 52 m)*, and the frame-0 phantom check in every matched-speed scenario |
| Phantom AEB after a range re-seed | round 1 | `range_jump_during_steady_follow` — the −10 m step puts the apparent range at 42 m, one metre inside the boundary |
| DISENGAGE on an empty road with a valid ego state | round 1 | `empty_road_nominal`, `empty_road_dropout`, `source_loss_mid_run` |
| A mock lane centre hides a real lead | round 1 | `lane_error_with_closing_lead` at 30 m *(moved from 65 m, where 39 m of margin let a system ignore the obstacle for two seconds and still pass)* |
| Missed braking against a lead braking at 6 m/s² | round 2 | `lead_brakes_6mps2_ego20_at_16m`, `_at_20m`, `_at_26m` inside the measured contact region; `_at_32m` outside it; `ego15_at_15m` decided on clearance; `ego25_at_30m`, which **both** commits fail *(moved: the previous four cases braked the lead at t = 1.0 s, which lifted the whole family out of the failing region and made all four pass on the commit that collides)* |
| Missed braking against a stopped obstacle | rounds 1–2 | `stationary_15mps_at_15m`, `stationary_20mps_at_26m`, `stationary_25mps_at_42m` one metre inside the avoidable side of the boundary; `stationary_20mps_at_29m` just outside it *(moved: 25/45/60/75 m, 16–32 m clear of the boundary)* |
| Braking beyond comfort with nothing to correct, below the emergency threshold | round 2, undetected until now | Every case with `max_commanded_decel_mps2 = 3.0`: the whole constant-range family, `lead_brakes_gently_2mps2`, `lead_accelerates_away`, `cutin_moderate_at_25m`, `reid_during_steady_follow`, `range_jump_during_steady_follow`; plus `BAND_UNWARRANTED` in the sweep |
| Over-braking: full authority where the kinematics ask for a third of it | rounds 1–2 | `stationary_20mps_at_75m` (requirement 2.74 m/s², ceiling 4.6 m/s²), `stationary_20mps_at_40m`, `dropout_mid_approach`, `lead_brakes_gently_2mps2` |
| Self-sustaining loop: `steering_rate -> plan_accel -> lane_departure` | rounds 1–3 | Every scenario with `recover_within_frames`: the four inside-region `constant_range_*` cases, `lead_brakes_then_leaves_lane`, `cutin_close_at_12m`, `cutin_moderate_at_25m`, `empty_road_dropout`, `bend_mrm_and_recover`, `reid_during_steady_follow`, `range_jump_during_steady_follow` |
| Held or zeroed steering during a minimum-risk manoeuvre | round 3 | `bend_nominal`, `bend_mrm_and_recover` |
| ADAS-DEC-21: a latched command after the loop ends | build | `terminate_during_emergency` (lead braking from frame 0 at 20 m, the middle of the contact region), `source_loss_mid_run` |

Ordinary cases, which exist so that the guards above cannot be satisfied by
simply never intervening: `stationary_20mps_at_40m`, `constant_range_52m`,
`constant_range_70m`, `lead_brakes_gently_2mps2`, `lead_accelerates_away`,
`cutin_moderate_at_25m`, `dropout_mid_approach`, `empty_road_nominal`,
`bend_nominal`.

## 7. Baseline: measured against both broken commits

Spec version 1.0, 33 scenarios. The library is backtested against the two
committed arbiter versions with independently known failures, because a harness
that cannot reproduce a failure it was told about is not a harness. Both runs
are in detached worktrees; the harness source is identical in both.

| | `25e3ba5` (phantom braking) | `1ce4886` (missed braking, HEAD's arbiter) |
|---|---|---|
| Result | **9 passed, 24 failed** | **7 passed, 26 failed** |
| `collided` | 1 | **6** |
| `clearance` | 4 | **10** |
| `late_intervention` | 0 | 1 |
| `phantom_intervention` | **19** | 9 |
| `over_braked` | 10 | 8 |
| `disproportionate_brake` | 6 | 4 |
| `failed_to_recover` | **10** | 1 |
| `slow_recovery` | 0 | 5 |
| `over_authority_state` | 8 | 2 |
| `speed_loss` | 1 | 0 |
| `disengaged` | 1 | 1 |
| `exit_released_brake` | 0 | 1 |

**The collision half of the specification now fires.** The previous version of
this library never emitted `collided`, `clearance`, `late_intervention`,
`missed_intervention` or `no_response` against either commit; it emitted only
phantom and recovery findings. It now emits seventeen collision-side findings
against `1ce4886` alone.

The two commits are separated in the direction each is broken, which is the
property the library needs and did not have:

| Scenario | `25e3ba5` | `1ce4886` | What the split means |
|---|---|---|---|
| `constant_range_40m` | FAIL, 6.39 m/s² | PASS, 3.00 m/s² | 40 m is inside the phantom region and outside the missed-braking one |
| `constant_range_52m`, `_70m` | PASS | PASS | the clear controls, which must keep passing |
| `lead_brakes_6mps2_ego20_at_16m/20m/26m` | FAIL (clearance 0.34/0.40/1.56 m) | FAIL (**contact**) | inside the contact region on both counts, contact on one |
| `lead_brakes_6mps2_ego20_at_32m` | FAIL (phantom at frame 0) | PASS | just outside the contact region; a 6 m widening shows up here first |
| `stationary_15mps_at_15m`, `_20mps_at_26m`, `_25mps_at_42m` | PASS at the exact physical limit | FAIL (**contact**) | one metre inside the avoidable side of the boundary |
| `stationary_20mps_at_29m` | PASS, 3.15 m | FAIL, 1.79 m | decided on the 2.0 m clearance, not on contact |
| `stationary_20mps_at_75m` | FAIL, 6.46 m/s² | FAIL, 6.05 m/s² | both over-brake where 2.74 m/s² is required |
| `reid_during_steady_follow` (40 m) | FAIL, 6.39 m/s² | PASS, 3.00 m/s² | inside the phantom region at 40 m; at the old 52 m both passed |

The envelope sweep, run as the gate, agrees and adds the boundaries:

```
25e3ba5:  FAIL: BAND_UNWARRANTED=3,  COLLISION=5,  EARLY=18, LATE=4, PHANTOM=15
1ce4886:  FAIL: BAND_UNWARRANTED=19, COLLISION=21, EARLY=15, LATE=8, PHANTOM=7
```

Read the signatures rather than the totals: `25e3ba5` is phantom-heavy and
`1ce4886` is collision-heavy, which is exactly what the two commit messages
claim and what three rounds of tuning oscillated between.

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
* move a scenario off its boundary. Section 3a records where each boundary is
  and which case sits either side of it; a case moved 10 m out is a case
  deleted, whether or not it still appears in the report;
* delete a scenario that guards a historical failure. Add to the table in
  section 6 instead — that table is the record of what this codebase has already
  got wrong once.

### Re-measuring a boundary

Boundaries are measurements, so they go stale when the plant changes. To
re-measure, in a detached worktree of the commit in question with the harness
copied in:

```bash
git worktree add --detach /tmp/bt <commit>
cp -r tests/scenarios /tmp/bt/tests/ && cp scripts/run_safety_sweep.py /tmp/bt/scripts/
ln -s "$PWD/models" /tmp/bt/models          # gitignored, needed for imports

cd /tmp/bt
PYTHONPATH=src:. python3 scripts/run_safety_sweep.py --profile dense \
    --ego 20 --range 12,16,20,26,30,32,40,44,52 --rate 0 --lead-decel 6

git worktree remove /tmp/bt                 # clean up
```

Then update the tables in section 3a, the `physics` strings that quote them, and
the `PROFILES` range axes in `tests/scenarios/sweep.py` so the grid still holds a
sample either side of the new boundary.
