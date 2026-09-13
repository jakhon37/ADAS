# Envelope sweep and real-footage justification analysis

Two tools that between them caught every longitudinal blocker the project has
had. They exist because single scenarios did not: the phantom AEB, the missed
braking and the false "0 frames at brake = 1.00" claim were each found by
sweeping an envelope or by reading a real run frame by frame, never by a
hand-written case.

**The sweep is the primary longitudinal safety gate.** The hand-picked scenario
library in `tests/scenarios/library.py` is the readable specification and it is
enforced under pytest, but it missed both of this module's historical failures
and the sweep found both. So `scripts/run_safety_sweep.py --gate` is what a
change has to pass; `docs/SAFETY_SPEC.md` is what explains why.

| | what it answers | needs a GPU |
|---|---|---|
| `scripts/run_safety_sweep.py` | over the whole operating envelope, *where* does the arbiter brake when it should not, and fail to brake when it must? | no |
| `scripts/analyze_run.py` | on real footage, is every intervention justified *by the scene the arbiter actually measured*? | only to record; not to analyse |

They share `tests/scenarios/instrument.py`, which captures the arbiter's
internal decision inputs without modifying it, and they share the physics in
`tests/scenarios/plant.py` and `tests/scenarios/oracle.py` with the scenario
suite, so there is one specification rather than two.

---

## 1. The sweep

### Running it

```bash
# THE GATE. 96 cells, ~11 s on the Xavier NX. Non-zero exit on any collision,
# miss, phantom, early or late intervention, or sub-emergency band fault.
PYTHONPATH=src:. python3 scripts/run_safety_sweep.py --gate

# same, with the committed artifact for diffing against the previous revision
PYTHONPATH=src:. python3 scripts/run_safety_sweep.py --gate \
    --json build/safety_sweep_gate.json

# the default envelope: 1260 cells, ~2.5 min on the Xavier NX
PYTHONPATH=src:. python3 scripts/run_safety_sweep.py --json /tmp/sweep.json

# the dense envelope, for finding a boundary precisely by hand
PYTHONPATH=src:. python3 scripts/run_safety_sweep.py --profile dense

# or sweep one axis at whatever resolution you want
PYTHONPATH=src:. python3 scripts/run_safety_sweep.py --ego 15 \
    --range 12,15,18,20,22,25,28,30 --rate 0 --lead-decel 0
```

`--gate` overrides `--profile` and `--fail-on` so the gate cannot be quietly
weakened by adding a flag. It fails on

```
COLLISION, MISSED, PHANTOM, LATE, EARLY, BAND_UNWARRANTED
```

which is both directions: `COLLISION` and `MISSED` are the passive one,
`PHANTOM`, `EARLY` and `BAND_UNWARRANTED` the aggressive one, and `LATE` is
intervening after the last moment the vehicle could still stop with room. A gate
that omitted either direction is how this arbiter came to oscillate between them.

No GPU, no TensorRT, no camera, no network. Deterministic: the same arguments
produce the same bytes, every time — verified by running the gate twice and
diffing both the text and the JSON. `--json` writes every cell, so the artifact
can be committed and two revisions of the arbiter diffed against each other.

### The three profiles, and why their range axes are what they are

| profile | cells | axes | time |
|---|---|---|---|
| `fast` (the gate) | 96 | ego {15,20,25} × range {12,16,26,32,40,44,52,70} × rate {0,−8} × lead decel {0,6} | ~11 s |
| `standard` | 1260 | ego {5..30 step 5} × range {5,8,12,16,20,26,30,32,40,43,44,52,60,70,80} × rate {+3,0,−1,−2,−4,−8,−15} × lead decel {0,6} | ~2.5 min |
| `dense` | 9900 | halved ego and range steps, 30 ranges, lead decel {0,3,6} | ~20 min |

A grid whose sample points all sit inside a failure region reports its width as
the width of the grid, and one whose points all sit outside reports it as zero.
Both are wrong, and the first version of the scenario library made exactly the
second mistake. So every profile keeps **at least one range either side of every
boundary this envelope is known to contain**:

| boundary | measured on | value | straddled by |
|---|---|---|---|
| constant-range phantom | `25e3ba5` | 26 m at 15 m/s, 43 m at 20 m/s, 66 m at 25 m/s | 26/32 · 40/44 · 52/70 |
| avoidable contact, lead braking at 6 m/s² | `1ce4886` | 16–26 m at 20 m/s (full stack) | 12/16 · 26/32 |
| stopped-obstacle avoidability = the plant's own stop distance | plant | 14.70 m at 15 m/s, 25.85 m at 20 m/s, 40.13 m at 25 m/s | 12/16 · 26/32 · 40/44 |

Every profile still contains `(rate = 0, decel = 0)` — the constant-range scene
that fix round 1 phantom-braked for — and `(rate = 0, decel = 6)` — the braking
lead that fix round 2 collided with.

### What each cell is

A single in-path lead, projected through `plant.render_box` so that the box the
arbiter's image-space in-path gate sees is geometrically consistent with the
range it is also given. The arbiter is handed
`ControlCommand(throttle=0.4, brake=0.0)` every frame: **a cruise request with
no brake in it at all**, so any brake in the output is the arbiter's own.

Each cell is run twice:

* **open loop** — the ego holds its speed and the lead follows its script, so
  the scene evolves identically whatever the arbiter decides. This isolates the
  *decision*, and it is what detects PHANTOM, EARLY, LATE and MISSED.
* **closed loop** — the ego obeys the arbitrated command through
  `tests.scenarios.plant.Plant`, brake rise time and all. This is what detects
  COLLISION, and it is the pass that would have caught fix round 2: an arbiter
  can look merely LATE open loop and still put the car into the lead once the
  plant is in the circuit.

### The physics the verdicts come from

Nothing is read from `ArbiterLimits`. Three instants, all from
`tests/scenarios/oracle.py`:

* **warrant** — the first frame at which the situation can no longer be
  recovered by `COMFORT_DECEL_MPS2` (3.0 m/s², 0.3 g) while preserving
  `REQUIRED_CLEARANCE_M` (2.0 m). Before it, an emergency intervention is not
  warranted: ordinary following control still has the situation. Computed under
  the oracle's *causal* assumption — the lead holds its current acceleration —
  so it never demands clairvoyance.
* **mandate** — the last frame from which full braking **through the real
  plant, including its 0.15 s brake rise**, still preserves that 2 m clearance.
  This is the last defensible moment to intervene, and it is a fact about the
  vehicle rather than a tuning constant.
* **lost** — the first frame from which even full braking through the real
  plant cannot avoid contact. Past it the collision is arithmetic.

Splitting them this way is deliberate: `warrant` is idealised kinematics because
it asks what the *situation* demands; `mandate` and `lost` run the actual
actuator because they ask what the *car* can still do.

An intervention counts as an emergency when the arbitrated deceleration reaches
`EMERGENCY_DECEL_MPS2` (3.5 m/s², half a metre above comfort so a
comfort-limited ramp cannot be mistaken for an AEB) or when the state is
`min_risk_maneuver` or `disengage`.

### The verdicts

| code | grid | meaning |
|---|---|---|
| `CORRECT` | `.` | intervened inside `[warrant, mandate]`, or correctly did nothing |
| `COLLISION` | `X` | the closed-loop pass put the ego into the lead. Overrides every open-loop opinion |
| `PHANTOM` | `P` | emergency-grade braking on a scene where **no emergency ever arises**, however long you wait |
| `EARLY` | `E` | emergency-grade braking before `warrant`. Not a loss of safety; an unnecessary emergency, which transfers risk to the vehicle behind |
| `LATE` | `L` | intervened after `mandate` — the car can no longer stop with 2 m to spare |
| `MISSED` | `M` | an emergency arose inside the window and no emergency-grade intervention followed |
| `INFEASIBLE` | (blank) | the implied lead speed is negative: head-on, outside a following ODD |

### Three independent gradings

The verdict above is one of three. Each is a different question about the same
cells, and a cell can be right about one and wrong about another.

| grading | field | values |
|---|---|---|
| emergency | `verdict` | `CORRECT` / `COLLISION` / `PHANTOM` / `EARLY` / `LATE` / `MISSED` / `INFEASIBLE` |
| headway | `headway_verdict` | `SOFT_OK` / `SOFT_PHANTOM` / `SOFT_MISSED` |
| sub-emergency band | `band_verdict` | `BAND_OK` / `BAND_UNWARRANTED` |

The **headway** requirement is graded against an RSS-style safe following gap.
Following at 5 m at 30 m/s with a matched speed is unsafe headway; the correct
answer is to open the gap, not to brake at full authority. Conflating the two is
exactly what produced the phantom, so a gentle deceleration is never counted as
an emergency failure.

The **sub-emergency band** is 3.0–3.5 m/s²: above `COMFORT_DECEL_MPS2`, below
`EMERGENCY_DECEL_MPS2`. It was added after the backtest of the scenario library
found it structurally invisible to everything else. The emergency grading starts
at 3.5 m/s², so 3.4 m/s² is not an intervention to it; the headway grading only
asks whether *some* response happened, so 3.4 m/s² satisfies it. Between them a
system can brake harder than any passenger would tolerate, for a whole run,
against a lead that never moved, and score `CORRECT` / `SOFT_OK` on every cell.
That is not hypothetical — `1ce4886` holds 3.0–3.3 m/s² on a constant-range
follow at every gap from 26 m to 44 m at 20 m/s, and 248 of its 1170 graded
cells are `BAND_UNWARRANTED`.

A cell is `BAND_UNWARRANTED` when a command in the band, held for at least
`band_frames_min` frames (3, the plant's own brake rise time, so a ramp edge is
not mistaken for a hold), first appears on a frame at which **neither** an
emergency has arisen (`warrant`) **nor** the headway has become unsafe. Nothing
in the scene asks for more than comfort at that instant, so the excess is the
system's own. Braking firmly to open a genuinely short gap is `BAND_OK`.

The band is closed at the bottom: a command of exactly 3.0 m/s² counts. The
comfort limit is the largest deceleration a headway law may use, so spending all
of it when there is no headway deficit is already the fault.

### Reading the output

Read it from the bottom up. The **REGIONS** section is the useful part: a count
tells you how bad it is, a boundary tells you what is wrong.

```
PHANTOM for every in-path range <= 20 m at ego 20 m/s (rate +0, lead_decel 0); correct from 25 m out
```

A statement that says *"reaches the top of the swept range axis, the real
boundary is beyond 80 m"* means the failure region runs off the edge of the grid
and you should widen the axis before believing the extent.

The **GRIDS** section prints one table per `(rate, lead decel)` slice, rows by
range and columns by ego speed, so a failure region shows up as a contiguous
block of one letter. That shape is the diagnosis; the count is not.

A second set of grids, and a second **SUB-EMERGENCY BAND REGIONS** section,
print the band grading the same way with `B` for `BAND_UNWARRANTED`. Reading the
two together is the point. On `25e3ba5` at ego 20 m/s with a constant range, the
gate prints:

```
  relative rate +0 m/s, lead braking at 0 m/s^2      SUB-EMERGENCY BAND
  range \ ego |    15    20    25                   range \ ego |    15    20    25
  -------------------------------                    -------------------------------
          70  |     .     .     .                            70  |     .     .     .
          52  |     .     .     P                            52  |     .     .     .
          44  |     .     .     P                            44  |     .     B     .
          40  |     .     P     P                            40  |     .     .     .
          32  |     .     P     P                            32  |     .     .     .
          26  |     P     P     P                            26  |     .     .     .
          16  |     P     P     P                            16  |     .     .     .
          12  |     P     P     P                            12  |     .     .     .
```

The emergency phantom stops at 43 m and the band picks up at 44 m. Without the
second grid the 44 m cell reads as `CORRECT`, and the boundary looks like a
cliff rather than what it is: full authority becoming 3.2 m/s² of unwarranted
braking.

---

## 2. The real-footage analyser

### Recording (needs the GPU)

```bash
ssh jetson-nx 'flock /tmp/jetson-gpu.lock -c "cd ~/myspace/ADAS && \
    PYTHONPATH=src:. python3 scripts/analyze_run.py record \
        --out /tmp/run400.json --frames 400 --ego-speed 15"'
```

This drives the real pipeline (YOLOX + UFLD over
`Ultra-Fast-Lane-Detection-v2/example.mp4`) with the arbiter instrumented, and
writes one JSON record per arbitration. It is the only part that touches the
GPU, and it **must** run under `flock /tmp/jetson-gpu.lock`.

### Analysing (no GPU)

```bash
PYTHONPATH=src:. python3 scripts/analyze_run.py analyze /tmp/run400.json
PYTHONPATH=src:. python3 scripts/analyze_run.py analyze /tmp/run400.json \
    --all --window 20 --json /tmp/report.json
```

Every frame that intervenes — enters `LIMITED`, `MIN_RISK_MANEUVER` or
`DISENGAGE`, or commands a brake above the threshold, or demands any
deceleration of its own — is dumped with its evidence: the lead, its **raw**
measured range over the preceding N frames, whether the closing rate was
measured or inferred from the seeded prior, the range source, and whether the
ego lane was occupied at all by the arbiter's own geometry.

### Attribution comes first

The pipeline has three brakes in it and they are not the same thing:

| | |
|---|---|
| **DEMAND** | the deceleration the arbiter itself asked for, from `_required_decel` |
| **COMMAND** | the brake the controller handed in |
| **ACTUATED** | the brake that reached the wheels |

A brake at the wheels is not evidence about the arbiter, and an arbiter with no
brake at the wheels is not evidence of restraint. Every frame is therefore
labelled:

* `ARBITER_BRAKE` — the arbiter raised the brake above the one it was handed.
* `ARBITER_DEMAND_ABSORBED` — the arbiter demanded deceleration of its own but
  the incoming command already exceeded it, so the pedal shows nothing. **This
  is the one that is easy to miss**: the demand is a real arbiter decision that
  only failed to show because the planner was braking anyway.
* `ARBITER_STATE_ONLY` — a degraded state with no braking of its own.
* `PASSTHROUGH` — the brake came from the planner and the arbiter neither added
  to it nor objected. Evidence about the planner, not about the arbiter, and
  never counted against it.

### How "justified" is decided

From the **raw tracker range history**, which is the measurement channel — never
from the arbiter's filtered range or its own closing rate, since those are the
things under test. A least-squares slope over the window gives an observed
closing rate *and its standard error*; a slope within two standard errors of
zero is range noise, not a measured closure, and is treated as no measurement at
all. That number plus the range and the ego speed goes into
`oracle.required_decel_mps2`, and the arbiter's own deceleration is compared
against it with the oracle's justification margin (50 % above the theoretical
minimum plus 0.5 m/s²).

| verdict | meaning |
|---|---|
| `JUSTIFIED` | the measured scene needs more than comfort braking |
| `JUSTIFIED_GRADED` | a graded response no larger than the measured closure demands |
| `JUSTIFIED_HEADWAY` | a soft response on a gap below the safe following distance |
| `JUSTIFIED_RANGE_ALONE` | the rate was not measurable, but the range is short enough that even a *stationary* obstacle there would demand more than comfort braking |
| `JUSTIFIED_DEGRADED` | perception or ego speed was unhealthy: a health response, not a traffic one |
| `NON_TRAFFIC_STATE` | a degraded state with no hazard flag, no AEB flag and no braking demand — health, kinematics, or a latched state |
| `PASSTHROUGH_PLANNER_BRAKE` | not the arbiter's action |
| `UNJUSTIFIED_NO_TARGET` | nothing was in the ego path at all |
| `UNJUSTIFIED_NOT_CLOSING` | there is a lead, but its measured range was flat or opening. **The phantom signature** |
| `UNJUSTIFIED_INFERRED_RATE` | emergency braking whose only support is the seeded `-ego_speed` prior, at a range where a stationary obstacle would not have demanded it |
| `UNJUSTIFIED_OVERREACTION` | a real but mild closure answered disproportionately |
| `UNJUSTIFIED_NO_HAZARD` | a safe headway, a negligible closure, and an intervention anyway |
| `INDETERMINATE` | too little history to say; reported, never counted either way |

And the inverse, which matters just as much: every frame that did **not**
intervene is checked against the same physics, and one whose measured scene
demanded more than comfort braking is counted as a `MISSED_REACTION`.

**Caveat the tool prints itself.** On the bundled clip the ego speed is
*simulated* while the footage came from a real vehicle, so absolute TTC figures
are not physical. What is exactly valid, and is the point, is the internal
consistency question: given the range history the arbiter was handed and the ego
speed it itself believed, does its own action follow?

---

## 3. The instrumentation

`tests/scenarios/instrument.py` wraps six methods on the `SafetyArbiter` *class*
for the lifetime of a context manager and unwinds them exactly on exit. It does
not modify `arbiter.py`, and it wraps the class rather than an instance so that
an arbiter built deep inside `ADASPipeline` is captured too — the only way to
instrument a real video run without editing the pipeline.

```python
from tests.scenarios.instrument import ArbiterInstrument

with ArbiterInstrument() as inst:
    run_whatever_calls_the_arbiter()
for rec in inst.frames:
    print(rec.frame_index, rec.demanded_decel_mps2, rec.brake_in, rec.brake_out)
```

| hook | captured |
|---|---|
| `arbitrate` | frame boundary, context summary, state, violations, reason |
| `_in_path` | per-track in-path verdict — *was the lane occupied at all?* |
| `_assess_lead` | the whole lead assessment: range, rate, whether the rate was measured, source, coasting |
| `_classify_hazard` | the hazard verdict and the findings it raised |
| `_required_decel` | the **demand** |
| `_synthesise_command` | **command in** versus **actuated out** |

**Every hook is optional.** The arbiter is about to be redesigned; a method that
no longer exists is recorded in `ArbiterInstrument.missing_hooks` and the rest of
the capture still works. A recording carries that list, and the analyser prints
it as a warning, so a half-empty capture can never be quietly mistaken for a
clean result.

---

## 4. Measured results against both broken commits

Jetson Xavier NX, JetPack 5.1.6, Python 3.8.10. Reported verbatim. Both commits
were run in detached worktrees with an identical copy of the harness, so the
only difference between the two columns is `src/adas/control/arbiter.py`:

* `25e3ba5` — "Fix round 1: three safety blockers fixed, three new ones found",
  arbiter md5 `d85a8a2b`. **Phantom braking.**
* `1ce4886` — "Arbiter round 2: phantom AEB removed, missed-braking regression
  introduced", arbiter md5 `d517e2aa`. **Missed braking.** This is HEAD's
  arbiter.

### The gate, `--gate` (96 cells, ~11 s each)

```
25e3ba5:  FAIL: BAND_UNWARRANTED=3,  COLLISION=5,  EARLY=18, LATE=4, PHANTOM=15
1ce4886:  FAIL: BAND_UNWARRANTED=19, COLLISION=21, EARLY=15, LATE=8, PHANTOM=7
```

Both exit 1. Read the signature rather than the total: `25e3ba5` is
phantom-heavy and `1ce4886` is collision-heavy, which is precisely what the two
commit messages claim and what three rounds of tuning oscillated between. A
ten-second grid separates them, because its eight ranges were chosen to straddle
the boundaries rather than to cover the axis evenly.

### `--profile standard`, 1260 cells (1170 graded, 90 infeasible)

| | `25e3ba5` | `1ce4886` |
|---|---|---|
| `COLLISION` | 96 (8.2%) | **235 (20.1%)** |
| `PHANTOM` | **98 (8.4%)** | 52 (4.4%) |
| `LATE` | 16 (1.4%) | 47 (4.0%) |
| `EARLY` | **333 (28.5%)** | 261 (22.3%) |
| `CORRECT` | 627 (53.6%) | 575 (49.1%) |
| `BAND_UNWARRANTED` | 90 | **248** |
| `SOFT_PHANTOM` | 88 | 88 |
| fired on a seeded, unmeasured rate | 431 | 149 |

### Boundary 1 — the constant-range phantom, `25e3ba5`

```
PHANTOM for every in-path range <=  8 m at ego  5 m/s (rate +0, lead_decel 0); correct from 12 m out
PHANTOM for every in-path range <= 12 m at ego 10 m/s (rate +0, lead_decel 0); correct from 16 m out
PHANTOM for every in-path range <= 26 m at ego 15 m/s (rate +0, lead_decel 0); correct from 30 m out
PHANTOM for every in-path range <= 43 m at ego 20 m/s (rate +0, lead_decel 0); correct from 44 m out
PHANTOM for every in-path range <= 60 m at ego 25 m/s (rate +0, lead_decel 0); correct from 70 m out
```

Resolved to a metre with `--ego 15,20,25 --range 26,27,...`: the boundary is
**26/27 m at 15 m/s, 43/44 m at 20 m/s, 66/67 m at 25 m/s** — a clean
`range ≈ 2.6 × ego speed`, one point per speed, no hysteresis. Closed loop
through the full stack the ego is dragged from 20 m/s down to 13.80, 14.83,
15.86 and 17.27 m/s at gaps of 12, 20, 30 and 40 m, and is untouched at 52 m and
70 m.

The same measurement on `1ce4886` gives emergency authority only up to 20 m at
20 m/s — and then a sustained 3.0–3.3 m/s² all the way out to 45 m, which is the
`BAND_UNWARRANTED` region and which nothing in this project measured before.

### Boundary 2 — avoidable contact against a lead braking at 6 m/s²

Sweep, `rate +0, lead_decel 6`, both commits:

```
25e3ba5:  COLLISION at ego 30 m/s for ranges 12, 16, 20, 26            (none at ego <= 25)
1ce4886:  COLLISION at ego 20 m/s for ranges 12, 16, 20, 26, 30, 32
          COLLISION at ego 25 m/s for ranges 16, 20, 26, 30, 32, 40, 43, 44, 52, 60
          COLLISION for range 12-80 at ego 30 m/s -- reaches the top of the swept range axis
```

Resolved to a metre in the closed loop through the full stack (planner →
controller → arbiter, arbiter's command actuated), with the lead braking from
**frame 0** rather than from t = 1.0 s:

| d₀ | 14 | 16 | 17 | 20 | 26 | 27 | 28 | 30 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| `1ce4886` min gap | +1.01 | **−0.00** | **−0.29** | **−0.65** | **−0.12** | +0.28 | +0.38 | +1.50 | +2.22 |
| `25e3ba5` min gap | +2.79 | +0.34 | +0.31 | +0.40 | +1.56 | +1.82 | +2.12 | +2.59 | +3.11 |

**Avoidable contact for d₀ = 16–26 m at 20 m/s.** The free second matters: with
the lead braking from t = 1.0 s instead, `1ce4886` collides at 15/20/25/30 m in
the earlier report but the *scenario* form of that case passed, because the
extra second of matched speed let the headway law settle first. Brake the lead
from frame 0 and the region is unambiguous.

### Boundary 3 — the stopped obstacle

The boundary here is the vehicle's own stop distance, measured by driving the
plant at `brake = 1.0`: **6.67 m at 10 m/s, 14.70 m at 15 m/s, 25.85 m at
20 m/s, 40.13 m at 25 m/s**. Closer than that, contact is arithmetic and no
arbiter can be blamed for it. Minimum true gap, full stack, closed loop:

| ego, d₀ | best available | `25e3ba5` | `1ce4886` |
|---|---|---|---|
| 15 m/s, 15 m | +0.30 | +0.30 | **−0.43** |
| 20 m/s, 26 m | +0.15 | +0.15 | **−1.21** |
| 20 m/s, 27 m | +1.15 | +1.15 | **−0.21** |
| 20 m/s, 29 m | +3.15 | +3.15 | +1.79 |
| 25 m/s, 41 m | +0.87 | +0.87 | **−0.83** |
| 25 m/s, 42 m | +1.87 | +1.87 | +0.17 |

`1ce4886` therefore throws away between 1.4 m and 2.0 m of clearance that the
vehicle physically has, which is the difference between stopping and hitting at
exactly the ranges where it matters. Both commits go to full authority for a car
40 m and 75 m away, where 5.26 and 2.74 m/s² are all that is required.

### Boundary 4 — the sub-emergency band

New in this revision, so there is no historical figure to compare against.
`1ce4886`, `--profile standard`:

```
BAND_UNWARRANTED at ego 15 m/s (rate +0, lead_decel 0) for ranges 20, 26
BAND_UNWARRANTED at ego 20 m/s (rate +0, lead_decel 0) for ranges 26, 30, 32, 40, 43, 44
BAND_UNWARRANTED for range 16-80 at ego 10 m/s (rate +0, lead_decel 6) -- reaches the top
                 of the swept range axis, the real boundary is beyond 80 m
```

248 cells. The shape is the mirror image of the phantom's: where `25e3ba5`
applied full authority out to 43 m, `1ce4886` applies 3.0–3.3 m/s² out to 44 m,
and the second is only two-thirds less wrong than the first. It is the residue
of "fixing" a phantom by attenuating it rather than by removing its cause.

### Real footage, 400 frames, YOLOX + UFLD, simulated ego 15 m/s

```
state histogram   : {'nominal': 357, 'limited': 43}
brake histogram   : {'0.00': 333, '(0,0.10)': 17, '[0.10,0.25)': 11,
                     '[0.25,0.50)': 12, '[0.50,0.90)': 20, '>=0.90 FULL': 7}
max brake actuated: 1.000
max arbiter demand: 5.00 m/s^2
frames at brake >= 0.90: 7
frames the arbiter RAISED the incoming brake: 0
lead continuity   : 9 distinct lead track ids [7, 10, 16, 21, 23, 24, 28, 32, 37],
                    15 lead switches, 2 frames where the lead was lost after having one

INTERVENTIONS: 50
  attribution:  ARBITER_STATE_ONLY 36 | ARBITER_DEMAND_ABSORBED 7 | PASSTHROUGH 7
  verdicts:     NON_TRAFFIC_STATE 36 | PASSTHROUGH_PLANNER_BRAKE 7
                JUSTIFIED_RANGE_ALONE 5 | JUSTIFIED_GRADED 1
                UNJUSTIFIED_OVERREACTION 1
  TOTAL UNJUSTIFIED 1
MISSED REACTIONS: 0
```

Four findings, none of them softened:

1. **The fix report's "0 frames at brake = 1.00" is still wrong on this clip.**
   Seven frames actuate a brake at or above 0.90, and one reaches exactly 1.00.
   They are frames 165–169 and 173–174, all in state `limited`.

2. **But none of those seven is the arbiter's.** `frames the arbiter RAISED the
   incoming brake: 0` — the pedal came from the planner and controller every
   time, and the arbiter passed it through. The demand/command split is what
   makes that statement possible; the round-1 investigation could not make it and
   attributed the pedal to the arbiter.

3. **The arbiter's own demand still reaches 5.0 m/s² on an inferred rate.**
   On frames 165–168 it demands 5.0 m/s² against a lead whose *arbiter* rate is
   −12.1 → −10.0 m/s and whose `rate_is_measured` flag is `False`, i.e. the
   seeded prior. The raw range history over the same window is
   `10.4 8.5 7.9 7.7 9.6 9.9 7.9 8.6 9.0 8.0 7.6 7.4` — a −2.2 ± 1.0 m/s slope,
   not a −12 m/s closure. Four of those five are graded
   `JUSTIFIED_RANGE_ALONE` only because at 7.4 m and 12 m/s a *stationary*
   obstacle would be unavoidable anyway; f168 is not, and is reported as
   `UNJUSTIFIED_OVERREACTION`: a measured −3.05 ± 0.99 m/s closure at 7.1 m needs
   0.9 m/s², and the arbiter demanded 5.0. On the very next frame, once the rate
   became *measured*, it read −0.58 m/s and the demand collapsed from 5.00 to
   0.66. The demand was tracking the seed, not the scene.

4. **The lead identity thrashes and the state latches on nothing.** Nine
   distinct lead track ids and fifteen lead switches in 400 frames, and 33 of the
   43 `limited` frames carry **no finding at all** — 22 of them with the reason
   `limited | no_in_path_lead`. A degraded state with no live evidence behind it
   is a latch, and a latch that outlives its cause is how the previous rounds'
   feedback loops sustained themselves.

`MISSED REACTIONS: 0` is a real result on this clip, not a clean bill of health:
the clip contains no genuine hard closure, and the significance test correctly
refuses to manufacture one out of a ±1.5 m range jitter.

---

## 5. For the redesign

The redesign is judged against these two tools, unchanged. Concretely:

* `scripts/run_safety_sweep.py --gate` **returns 0.** That is the bar, in CI, on
  every change. It fails on `COLLISION`, `MISSED`, `PHANTOM`, `LATE`, `EARLY`
  and `BAND_UNWARRANTED`, i.e. on both error directions at once, which is the
  property no previous gate had.
* `--profile standard` with **zero** `COLLISION`, **zero** `MISSED`, **zero**
  `PHANTOM`, **zero** `BAND_UNWARRANTED`. `EARLY` and `LATE` should shrink to
  isolated cells at the grid's edges, not regions.
* `PYTHONPATH=src python3 -m pytest tests/test_scenarios.py` passing — the
  readable form of the same requirements, with the physics argument attached to
  each.
* On the recorded 400-frame run: no `UNJUSTIFIED_*` verdict, and no arbiter
  demand above comfort on a frame whose closing rate is not measured.

**Do not narrow the grid to make the gate pass.** The range axes in
`PROFILES` are chosen to straddle measured boundaries (section 1); removing 16 m
or 44 m from `fast` removes the only evidence that a boundary moved. If the
plant changes, re-measure the boundaries first (`docs/SAFETY_SPEC.md`, section 8)
and then move the axes to straddle the new ones.

The instrumentation is written to survive the rewrite: if a hooked method is
renamed, it is listed in `missing_hooks` and the surrounding capture still
works, but the fields it fed go empty. Keep something equivalent to
`_required_decel` (the demand) and `_synthesise_command` (command in versus out)
distinguishable, or re-point the hooks — losing that split is losing the only
way to attribute a brake.
