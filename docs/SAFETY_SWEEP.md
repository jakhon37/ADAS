# Envelope sweep and real-footage justification analysis

Two tools that between them caught every longitudinal blocker the project has
had. They exist because single scenarios did not: the phantom AEB, the missed
braking and the false "0 frames at brake = 1.00" claim were each found by
sweeping an envelope or by reading a real run frame by frame, never by a
hand-written case.

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
# CI resolution: 54 cells, ~7 s, fails the build on a collision or a phantom
PYTHONPATH=src:. python3 scripts/run_safety_sweep.py --profile fast \
    --fail-on COLLISION,MISSED,PHANTOM

# the default envelope: 840 cells, ~100 s on the Xavier NX
PYTHONPATH=src:. python3 scripts/run_safety_sweep.py --json /tmp/sweep.json

# the dense envelope, for finding a boundary precisely by hand
PYTHONPATH=src:. python3 scripts/run_safety_sweep.py --profile dense

# or sweep one axis at whatever resolution you want
PYTHONPATH=src:. python3 scripts/run_safety_sweep.py --ego 15 \
    --range 12,15,18,20,22,25,28,30 --rate 0 --lead-decel 0
```

No GPU, no TensorRT, no camera, no network. Deterministic: the same arguments
produce the same bytes, every time. `--json` writes every cell for diffing two
revisions of the arbiter against each other.

### The three profiles

| profile | cells | axes | time |
|---|---|---|---|
| `fast` | 54 | ego {10,20,30} × range {8,20,45} × rate {0,−2,−8} × lead decel {0,6} | ~7 s |
| `standard` | 840 | ego {5..30 step 5} × range {5,8,12,15,20,25,30,45,60,80} × rate {+3,0,−1,−2,−4,−8,−15} × lead decel {0,6} | ~100 s |
| `dense` | 6600 | halved ego and range steps, lead decel {0,3,6} | ~13 min |

`standard` reproduces the historical 336-cell envelope and adds the two ranges
(15 m and 25 m) that the round-2 collision report named, on both lead-braking
axes. Every profile contains `(rate = 0, decel = 0)` — the constant-range scene
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

The **headway** requirement is graded separately (`SOFT_OK`, `SOFT_PHANTOM`,
`SOFT_MISSED`) against an RSS-style safe following gap. Following at 5 m at
30 m/s with a matched speed is unsafe headway; the correct answer is to open the
gap, not to brake at full authority. Conflating the two is exactly what produced
the phantom, so a gentle deceleration is never counted as an emergency failure.

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

## 4. Measured results against the current arbiter

Commit `1ce4886` ("Arbiter round 2: phantom AEB removed, missed-braking
regression introduced"), Jetson Xavier NX, JetPack 5.1.6, Python 3.8.10.
Reported verbatim.

### Sweep, `--profile standard`, 840 cells (780 graded, 60 infeasible)

```
COLLISION   174  (22.3% of graded)
PHANTOM      52  ( 6.7% of graded)
LATE         26  ( 3.3% of graded)
EARLY       193  (24.7% of graded)
CORRECT     335  (42.9% of graded)
FAILURES    445
headway (graded separately): SOFT_OK 716, SOFT_PHANTOM 64
of the PHANTOM and EARLY cells, 149 fired on a frame where the closing rate
was still the seeded prior, not a measurement
```

**The phantom is not gone. It has a clean boundary at range ≈ ego speed × 1 s.**
With a lead holding a *constant* range and a constant *opening* range alike:

```
PHANTOM for every in-path range <=  5 m at ego  5 m/s ...; correct from  8 m out
PHANTOM for every in-path range <= 12 m at ego 10 m/s ...; correct from 15 m out
PHANTOM for every in-path range <= 15 m at ego 15 m/s ...; correct from 20 m out
PHANTOM for every in-path range <= 20 m at ego 20 m/s ...; correct from 25 m out
PHANTOM for every in-path range <= 25 m at ego 25 m/s ...; correct from 30 m out
PHANTOM for every in-path range <= 30 m at ego 30 m/s ...; correct from 45 m out
```

```
  relative rate +0 m/s, lead braking at 0 m/s^2
  range \ ego |     5    10    15    20    25    30
  -------------------------------------------------
          80  |     .     .     .     .     .     .
          60  |     .     .     .     .     .     .
          45  |     .     .     .     .     .     .
          30  |     .     .     .     .     .     P
          25  |     .     .     .     .     P     P
          20  |     .     .     .     P     P     P
          15  |     .     .     P     P     P     P
          12  |     .     P     P     P     P     P
           8  |     .     P     P     P     P     P
           5  |     P     P     P     P     P     P
```

Every one of these fires on frame 0 with the closing rate still seeded. Isolated:

```
$ python3 scripts/run_safety_sweep.py --ego 20 --range 20 --rate 0 --lead-decel 0
PHANTOM  ego 20.0  range 20.0  rate +0.0  lead_decel 0.0
         warrant=None mandate=None lost=None | hard=0 soft=0
         max_cmd_decel=5.00 max_demand=5.00
         first hard state limited; rate was INFERRED, not measured
         findings ttc_0.80s_below_warn_threshold, headway_20.0m_below_rss_45.3m,
                  aeb_deferred_unmeasured_rate_-20.0m/s_track1
```

A lead sitting at a rock-steady 20 m draws 5.0 m/s² on the first frame it is
seen, justified by a fabricated `-ego_speed` rate. Round 1's phantom commanded
brake 1.00; this one commands 0.62. It is attenuated, not removed.

**The missed braking is confirmed, and it is worse than "missed": it collides.**
The round-2 report said the arbiter collides against a lead braking at 6 m/s² at
ego 20 m/s from 15, 20, 25 and 30 m. The sweep found exactly that, independently:

```
COLLISION at ego 20 m/s (rate +0, lead_decel 6) for ranges 12, 15, 20, 25, 30
COLLISION for every in-path range <= 30 m at ego 20 m/s (rate -8, lead_decel 6); correct from 45 m out
COLLISION for every in-path range <= 45 m at ego 25 m/s (rate -8, lead_decel 6); correct from 60 m out
COLLISION for every in-path range <= 15 m at ego 20 m/s (rate -15, lead_decel 0); correct from 20 m out
```

```
  relative rate +0 m/s, lead braking at 6 m/s^2
  range \ ego |     5    10    15    20    25    30
  -------------------------------------------------
          80  |     .     .     .     .     .     X
          60  |     E     .     .     .     X     X
          45  |     E     .     .     L     X     X
          30  |     E     .     L     X     X     X
          25  |     E     .     L     X     X     X
          20  |     E     .     L     X     X     X
          15  |     E     .     .     X     X     X
          12  |     E     E     .     X     .     X
           8  |     E     .     .     .     .     .
           5  |     E     .     .     .     .     .
```

Two things to notice. The collision region is *not* bounded by short range: at
ego 30 m/s the arbiter collides from 80 m and clears at 8 m, the opposite of the
phantom's shape. And the `L` cells directly below the `X` band are the same
failure caught one grid step earlier — the arbiter intervenes, but after the last
frame from which the real actuator could still stop with 2 m to spare.

The 193 `EARLY` cells are the phantom's other face: on a scene where a real
hazard *does* eventually appear, the same seeded rate makes the arbiter brake at
emergency authority long before comfort braking has run out. At ego 5 m/s with a
lead closing at 4 m/s, this happens at every range up to 45 m.

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

The redesign is judged against these two tools, unchanged. Concretely, a
redesigned arbiter should reach:

* `--profile standard` with **zero** `COLLISION`, **zero** `MISSED`, **zero**
  `PHANTOM`. `EARLY` and `LATE` should shrink to isolated cells at the grid's
  edges, not regions.
* `--profile fast --fail-on COLLISION,MISSED,PHANTOM` returning 0, in CI.
* On the recorded 400-frame run: no `UNJUSTIFIED_*` verdict, and no arbiter
  demand above comfort on a frame whose closing rate is not measured.

The instrumentation is written to survive the rewrite: if a hooked method is
renamed, it is listed in `missing_hooks` and the surrounding capture still
works, but the fields it fed go empty. Keep something equivalent to
`_required_decel` (the demand) and `_synthesise_command` (command in versus out)
distinguishable, or re-point the hooks — losing that split is losing the only
way to attribute a brake.
