"""Deterministic longitudinal (and minimal lateral) vehicle plant, plus the
sensor model that stands between it and the system under test.

This module is the GROUND TRUTH of the acceptance harness.  Nothing in it is
derived from :mod:`adas.control.arbiter`; every constant below is a statement
about a vehicle, a road or a measured pipeline timing, and each one carries the
reasoning that produced it.  The oracle in :mod:`tests.scenarios.oracle` judges
the system against *this* model, so the model's honesty matters more than its
sophistication.

Modelling decisions, stated up front so that a reviewer can disagree with them
explicitly rather than discover them in the arithmetic:

* **Point masses on a straight, flat, dry road.**  No grade, no rolling
  resistance, no aerodynamic drag.  Over the 5-20 s that a longitudinal safety
  scenario lasts, drag on a passenger car is under 0.4 m/s^2 and grade is zero
  by assumption; both are small next to the 8 m/s^2 braking authority and
  omitting them keeps the oracle's closed-form kinematics exact.
* **Bumper-to-bumper gap.**  ``gap_m`` is the clear distance between the ego's
  front bumper and the lead's rear bumper.  Vehicle lengths therefore never
  appear: ``gap_m <= 0`` IS contact.
* **First-order actuator lag.**  A commanded brake does not produce its
  deceleration instantly.  This is the difference between a system that
  "intervenes in time" on paper and one that does not on the road, so it is
  modelled rather than assumed away.
* **No wall clock, no unseeded randomness.**  Every run is reproducible bit for
  bit.  Noise is drawn from a counter-based stream keyed on
  ``(seed, frame, stream, object)``, so it does not depend on how many draws
  happened earlier, on iteration order, or on how many objects are in the scene.

The plant is deliberately *more* capable than the vehicle the system believes
it is driving in one respect only: it never refuses a command.  Saturation and
authority limits are applied here, in the plant, so a system that asks for 20
m/s^2 gets 8 m/s^2 and the harness can say so.

What this module models that a naive simulator does not
-------------------------------------------------------
Each of the following was added because a backtest of the harness against two
committed arbiter versions with *known measured* failures showed the harness
could not reproduce the failure at all.  A simulated world that cannot express
the defect cannot be a specification against it.

1. **Sensor noise and bias** (:attr:`PerceptionSpec.range_noise_m`,
   :attr:`PerceptionSpec.range_bias_frac`, :attr:`PerceptionSpec.box_noise_px`,
   :attr:`PerceptionSpec.lateral_noise_m`).  Seeded, reproducible, and
   requestable in one line via :func:`noisy_perception`.
2. **Sense-to-act latency** (:data:`SENSE_LATENCY_S`).  The observation handed
   to the system describes the world as it was one frame ago, because that is
   how long this board takes to turn photons into a command.
3. **An honest range rate.**  A camera does not measure closing speed; it
   differentiates range.  The reported ``velocity_mps`` is a least-squares slope
   over the track's own measured range history, so range noise becomes rate
   noise and a re-identified track has *no* rate until it has re-accumulated
   one.  Reporting the true closing rate on a brand-new track, as this module
   used to, made every re-identification scenario vacuous.
4. **Real in-path geometry** (:attr:`ObjectState.in_ego_lane`,
   :attr:`PerceptionSpec.lane_offset_error_m`).  ``in_ego_lane`` is computed
   from where the object actually is, through a lane estimate that a scenario
   may deliberately shift, so a correct in-path decision is distinguishable from
   a lucky one.
5. **Many objects at once** (:class:`Plant` ``others``).  Including out-of-lane
   ones, which must not be braked for, and simultaneous hazards.
6. **A visible throttle/brake conflict** (:attr:`WorldState.pedal_conflict`).
   The plant used to compute ``2.5*throttle - 8.0*brake``, which silently netted
   a simultaneous full throttle and full brake into a plausible-looking
   -5.5 m/s^2.  It now applies brake override, like a real car, and *records*
   that the two were commanded together so the judgement layer can fail on it.
7. **Variable frame period** (:attr:`PlantConfig.dt_schedule`).  A dropped or
   doubled frame is the commonest real perturbation of a 20 Hz pipeline, and
   this pipeline's own measured frame interval has a 174 ms maximum against a
   58.6 ms median.
8. **The real tracker, optionally, in the loop**
   (:attr:`PerceptionSpec.use_real_tracker`).  Off by default: clean injected
   tracks isolate the decision layer and run fast.  On, the plant renders
   detections and drives :class:`adas.tracking.MultiObjectTracker`, so Kalman
   range filtering, Hungarian association, M-of-N confirmation and the tracker's
   own ego-lane decision are all exercised.  Still CPU-only: the tracker imports
   numpy and nothing else.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from adas.core.models import (
    BoundingBox,
    EgoState,
    LaneLine,
    LaneModel,
    PerceptionStatus,
    TrackedObject,
)

# --------------------------------------------------------------------------- #
# Physical constants.  Each one is a claim about the world, not about the code.
# --------------------------------------------------------------------------- #

DT_S = 0.05
"""Nominal simulation timestep, seconds (20 Hz).

Chosen to match the pipeline's nominal frame period so that one plant step is
one perception frame and "frame N" means the same thing to the plant, the
oracle and the report.  It is the *nominal* period only: see
:attr:`PlantConfig.dt_schedule` for per-frame overruns.  The harness never reads
a clock, so a slow CI machine and a fast one produce identical results.
"""

MAX_BRAKE_DECEL_MPS2 = 8.0
"""Maximum braking authority, m/s^2.

Tyre-road friction limit on dry asphalt: a = mu * g with mu ~= 0.82 and
g = 9.81 gives 8.0 m/s^2.  This is the number the oracle uses when it asks
"could this collision have been avoided?", so it is deliberately the *optimistic*
end of a real car's capability -- a harness that assumed a weaker vehicle would
excuse late braking that a real car could have survived.
"""

MAX_ACCEL_MPS2 = 2.5
"""Maximum forward acceleration, m/s^2.

A mid-size passenger car at part throttle: 0-100 km/h in roughly 11 s.  The ego
never needs more than this in any scenario here; it exists so that recovery
after an intervention takes a realistic amount of time instead of being
instantaneous.
"""

BRAKE_RISE_TIME_S = 0.15
"""Time from a brake demand to 90% of the demanded deceleration, seconds.

Hydraulic pressure build-up plus pad bite.  Euro NCAP AEB test protocols and
ISO 15622 both work with 0.1-0.2 s for a modern electro-hydraulic system; 0.15 s
is the middle of that band.  Modelled as a first-order lag, so the time
constant is ``BRAKE_RISE_TIME_S / ln(10)``.
"""

THROTTLE_RISE_TIME_S = 0.35
"""Time from a throttle demand to 90% of the demanded acceleration, seconds.

Powertrain torque response is slower than brake response.  This only affects
how quickly the ego recovers speed after an intervention, which is exactly what
the recovery assertions measure, so it is modelled rather than idealised.
"""

BRAKE_OVERRIDE_PEDAL = 0.02
"""Brake pedal fraction at or above which the throttle request is cut.

Every production drivetrain implements brake override: press the brake and the
engine torque request goes to zero, irrespective of the throttle.  0.02 of full
brake is 0.16 m/s^2, about the threshold at which a pedal switch closes.

This constant is what stops the plant from *hiding* a simultaneous throttle and
brake.  The previous model computed ``2.5*throttle - 8.0*brake``, so a command
of full throttle AND full brake -- an unambiguous control fault -- integrated as
a perfectly ordinary -5.5 m/s^2 and left no trace anywhere.
"""

WHEELBASE_M = 2.8
"""Front-to-rear axle distance, metres.  Mid-size passenger car."""

MAX_ROAD_WHEEL_RAD = 0.436
"""Road-wheel angle at full steering command, radians (25 degrees).

The controller emits a steering command normalised to +/-1 against a 25 degree
full-scale, so a command of 1.0 means 25 degrees at the road wheel.  25 degrees
is a normal maximum for a passenger car at the road wheel (the steering wheel
turns much further; the ratio is not modelled).
"""

STEERING_RISE_TIME_S = 0.10
"""Time from a steering demand to 90% of the demanded road-wheel angle."""

CAMERA_FOCAL_PX = 910.0
"""Vertical focal length in pixels for a 1280x720 image.

Used only to render a plausible bounding box from the true range, so that the
system's in-path gate has something geometrically consistent to look at.  This
is the focal length the project's own camera default documents for this image
size; using a different one here would make the rendered boxes disagree with
every other consumer of a 1280x720 frame in the repo.
"""

CAMERA_HEIGHT_M = 1.2
"""Camera mount height above the road, metres.  Typical windscreen mounting."""

HORIZON_PX = 360.0
"""Row of the horizon in a 1280x720 image with a level camera: the centre row."""

CAR_HEIGHT_M = 1.5
"""Nominal height of a passenger car, metres.  Renders the box height."""

CAR_WIDTH_M = 1.8
"""Nominal width of a passenger car, metres.  Renders the box width and sets the
default lateral footprint used by the in-path test."""

EGO_HALF_WIDTH_M = 0.9
"""Half the ego's own width, metres.  Half of :data:`CAR_WIDTH_M`."""

LANE_WIDTH_M = 3.5
"""Lane width, metres.  The narrow end of a motorway lane (3.5-3.75 m), chosen
because a narrow lane makes the in-path test *harder* to pass, not easier."""

LANE_HALF_WIDTH_M = LANE_WIDTH_M / 2.0
"""Half a lane, metres.  The corridor that :attr:`ObjectState.in_ego_lane` uses."""

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720

LOOKAHEAD_M = 20.0
"""Distance ahead at which the lane centre is measured, metres.

Used to turn a metric lateral offset into a lane-centre column.  20 m is about
1 s of preview at motorway speed, which is the range a lane detector actually
fits over.
"""

SENSE_LATENCY_S = 0.055
"""Capture-to-command delay of the real pipeline on this board, seconds.

Measured, not assumed.  README.md's per-stage table for the 200-frame
YOLOX-Nano + UFLD-v2 run on this Xavier NX (2026-09-13, GPU mutex held) gives
the mean cost of every stage between the shutter and the actuator write:

===================== ======= =======
stage                 mean ms p95 ms
===================== ======= =======
detect (YOLOX-Nano)     12.68   15.32
lane (UFLD-v2)          41.17   46.82
track                    0.95    3.04
plan                     0.21    0.29
control                  0.08    0.10
arbitrate                0.46    0.97
**total**             **55.55** 66.54
===================== ======= =======

55.55 ms mean, 66.5 ms at the p95 sum -- which is the 45-60 ms end-to-end figure
the safety case quotes, measured.  Rounded onto the 50 ms frame grid by
:class:`Sensor` this is exactly **one frame** of delay at the nominal period,
and two once the frame period stretches past 110 ms.

The consequence is the point: the world the arbiter reasons about is one frame
stale, so at 20 m/s of closure it is reasoning about a gap that is already 1.0 m
smaller than it believes.  A harness with zero latency credits the system with
information it does not have, and every "it intervened in time" verdict taken
from it is worth 1 m less than it looks.
"""

ACTUATION_LATENCY_S = 0.0
"""Delay between a command being written and the actuator receiving it, seconds.

Zero by default, and that default is a claim: the 55.6 ms measured above is
capture-to-*command*, and everything after the command -- hydraulic build-up and
pad bite -- is already modelled explicitly by :data:`BRAKE_RISE_TIME_S` rather
than as dead time.  Modelling it twice would make the plant a weaker vehicle
than the oracle assumes and would excuse late braking.

The knob exists because a CAN-bus or actuator-ECU queue is real dead time on
some vehicles, and a scenario that wants to ask "what if the bus adds a frame?"
should be able to.
"""


def _lag_alpha(dt_s: float, rise_time_s: float) -> float:
    """First-order lag coefficient reaching 90% of a step in ``rise_time_s``.

    A first-order lag reaches 90% after ``ln(10) * tau``, so ``tau`` is
    ``rise_time_s / ln(10)`` and the per-step blend is ``1 - exp(-dt/tau)``.
    """
    if rise_time_s <= 0.0:
        return 1.0
    tau = rise_time_s / math.log(10.0)
    return 1.0 - math.exp(-dt_s / tau)


# --------------------------------------------------------------------------- #
# Deterministic noise
# --------------------------------------------------------------------------- #

_STREAM_RANGE = 1
_STREAM_LATERAL = 2
_STREAM_BOX_X = 3
_STREAM_BOX_Y = 4
_STREAM_LANE = 5
_STREAM_EGO_SPEED = 6


def _gauss(seed: int, frame: int, stream: int, index: int, sigma: float) -> float:
    """A zero-mean Gaussian draw that is a pure function of its coordinates.

    The harness's determinism test requires two runs of the same scenario to
    agree bit for bit, and the obvious implementation -- one
    :class:`random.Random` advanced once per draw -- satisfies that only as long
    as nothing changes the *order* or *number* of draws.  Adding a second object
    to a scene, or skipping a draw on a dropout frame, silently reshuffles the
    noise on every object after it and turns an unrelated edit into a
    reproducibility failure.

    So the stream is counter-based instead: the value at
    ``(seed, frame, stream, index)`` is fixed forever, independent of what else
    the run did.  Two scenarios that differ only in a third vehicle see
    bit-identical noise on the first two.

    Args:
        seed: The scenario's seed.
        frame: Frame index; noise is white across frames.
        stream: Which quantity is being perturbed (one of the ``_STREAM_*``
            constants), so range noise and lateral noise are independent.
        index: Object index, so two objects are perturbed independently.
        sigma: Standard deviation.  ``<= 0`` returns exactly 0.0, which keeps a
            noise-free scenario exactly noise-free rather than nearly so.

    Returns:
        The draw, in the units of ``sigma``.
    """
    if sigma <= 0.0:
        return 0.0
    key = (int(seed) & 0xFFFFFFFF) * 1000003
    key ^= (int(frame) & 0xFFFFF) * 2654435761
    key ^= (int(stream) & 0xFF) * 40503
    key ^= (int(index) & 0xFFFF) * 2246822519
    return random.Random(key & 0x7FFFFFFFFFFFFFFF).gauss(0.0, sigma)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlantConfig:
    """Vehicle and integration parameters.  Defaults are the constants above.

    Attributes:
        dt_s: Nominal timestep, seconds.
        dt_schedule: Per-frame timestep override.  ``dt_schedule[i]`` is the
            duration of the step that advances the world FROM frame ``i`` to
            frame ``i+1``; frames past the end of the tuple use ``dt_s``.  Build
            one with :func:`dt_with_overruns`.

            This exists because a 20 Hz vision pipeline does not run at 20 Hz.
            The measured frame interval of this board's own 200-frame run is
            58.56 ms at the median, 75.06 ms at p95 and 174.33 ms at the
            maximum, and the production log records ``Frame 325 took 187.7 ms,
            more than 3x the 50.0 ms nominal period``.  A frame that takes 187 ms
            is a frame during which the last command acted for 187 ms and the
            world moved 3.7 m at 20 m/s while the system was blind.  A harness
            that only ever steps 50 ms cannot see what that does.
        actuation_latency_s: Dead time between a command being issued and the
            actuator receiving it.  See :data:`ACTUATION_LATENCY_S`.
        brake_override_pedal: See :data:`BRAKE_OVERRIDE_PEDAL`.
    """

    dt_s: float = DT_S
    max_brake_decel_mps2: float = MAX_BRAKE_DECEL_MPS2
    max_accel_mps2: float = MAX_ACCEL_MPS2
    brake_rise_time_s: float = BRAKE_RISE_TIME_S
    throttle_rise_time_s: float = THROTTLE_RISE_TIME_S
    wheelbase_m: float = WHEELBASE_M
    max_road_wheel_rad: float = MAX_ROAD_WHEEL_RAD
    steering_rise_time_s: float = STEERING_RISE_TIME_S
    dt_schedule: Tuple[float, ...] = ()
    actuation_latency_s: float = ACTUATION_LATENCY_S
    brake_override_pedal: float = BRAKE_OVERRIDE_PEDAL

    def dt_for_frame(self, frame: int) -> float:
        """Duration of the step that advances the world from ``frame``."""
        if 0 <= frame < len(self.dt_schedule):
            return float(self.dt_schedule[frame])
        return float(self.dt_s)


DEFAULT_PLANT = PlantConfig()


def dt_with_overruns(
    frames: int,
    overruns: Dict[int, float],
    nominal_s: float = DT_S,
) -> Tuple[float, ...]:
    """A per-frame timestep schedule with named frames stretched.

    Args:
        frames: Length of the schedule.
        overruns: ``{frame_index: duration_s}``.  A duration, not a multiplier,
            so a scenario quotes the measured number directly:
            ``{325: 0.1877}`` is the frame from the production log.
        nominal_s: Duration of every other frame.

    Returns:
        A tuple suitable for :attr:`PlantConfig.dt_schedule`.

    Example:
        A 20 Hz run with one 3.75x overrun and one dropped frame (a doubled
        period, which is what a drop looks like to the consumer)::

            dt_with_overruns(400, {325: 0.1877, 120: 0.10})
    """
    out = [float(nominal_s)] * int(frames)
    for idx, dur in overruns.items():
        if 0 <= int(idx) < len(out):
            out[int(idx)] = float(dur)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Objects in the world
# --------------------------------------------------------------------------- #

#: A lead acceleration law.  Arguments are ``(t_s, lead_speed_mps,
#: ego_speed_mps, gap_m)`` and the return value is the lead's commanded
#: acceleration in m/s^2.  It is a *script*: the lead is ground truth and obeys
#: it exactly, with no actuator lag of its own.
LeadAccelFn = Callable[[float, float, float, float], float]

#: A lateral position law: ``t_s -> metres right of the true lane centre``.
LateralFn = Callable[[float], float]


@dataclass(frozen=True)
class LeadSpec:
    """A scripted vehicle in the world.

    Despite the name (kept because the whole harness refers to it) this
    describes ANY object, not only the followed lead: an out-of-lane vehicle in
    the next lane over is the same dataclass with a non-zero
    ``lateral_offset_m``.  :class:`Plant` takes one of these as the lead and any
    number of them as ``others``.

    Attributes:
        initial_gap_m: Bumper-to-bumper clear distance at t=0, metres.  May be
            negative for an object behind the ego, which is then never detected
            by a forward camera but still occupies the world.
        initial_speed_mps: Speed at t=0, m/s.
        accel_fn: The longitudinal acceleration law; see :data:`LeadAccelFn`.
        appears_at_s: Before this time the object is not in the scene at all
            (a cut-in).  It still travels its script, so the gap at the moment
            it appears is a physical consequence of the script, not a jump.
        vanishes_at_s: After this time the object has left the scene.  Used for
            the "the hazard is over, now recover" half of the recovery
            assertions.
        label: Human-readable description for the report, and the detection
            class label handed to the tracker ("car", "truck", ...).
        lateral_offset_m: Constant lateral position, metres right of the TRUE
            lane centre.  0.0 is dead ahead in the ego's lane;
            ``+LANE_WIDTH_M`` is the middle of the next lane to the right.
        lateral_fn: Optional lateral position law overriding
            ``lateral_offset_m``; see :func:`lateral_cut_in`.
        width_m: Lateral footprint, metres.  Sets the rendered box width and the
            in-path overlap test.
        height_m: Vertical extent, metres.  Sets the rendered box height, which
            is what a pinhole range estimator divides by -- so a truck at
            ``height_m=3.2`` handed to a 1.5 m height prior is mis-ranged by
            more than a factor of two, exactly as on the road.
    """

    initial_gap_m: float
    initial_speed_mps: float
    accel_fn: LeadAccelFn
    appears_at_s: float = 0.0
    vanishes_at_s: float = float("inf")
    label: str = ""
    lateral_offset_m: float = 0.0
    lateral_fn: Optional[LateralFn] = None
    width_m: float = CAR_WIDTH_M
    height_m: float = CAR_HEIGHT_M

    def present_at(self, t_s: float) -> bool:
        """Whether the object occupies the scene at ``t_s``."""
        return self.appears_at_s - 1e-9 <= t_s < self.vanishes_at_s

    def lateral_at(self, t_s: float) -> float:
        """Lateral position at ``t_s``, metres right of the true lane centre."""
        if self.lateral_fn is not None:
            return float(self.lateral_fn(t_s))
        return float(self.lateral_offset_m)

    @property
    def detection_label(self) -> str:
        """The class label a detector would emit for this object."""
        low = (self.label or "").lower()
        for known in ("truck", "bus", "motorcycle", "bicycle", "person", "car"):
            if known in low:
                return known
        return "car"


#: Readability alias.  A :class:`LeadSpec` that is not the lead is an object.
ObjectSpec = LeadSpec


def lead_stationary() -> LeadAccelFn:
    """A parked or stopped vehicle: zero speed, zero acceleration, forever."""

    def fn(t_s: float, lead_v: float, ego_v: float, gap_m: float) -> float:
        return 0.0

    return fn


def lead_constant_speed() -> LeadAccelFn:
    """Cruise: the lead holds whatever speed it started with."""

    def fn(t_s: float, lead_v: float, ego_v: float, gap_m: float) -> float:
        return 0.0

    return fn


def lead_brakes(decel_mps2: float, start_s: float = 0.0, stop_s: float = float("inf")) -> LeadAccelFn:
    """The lead brakes at a constant rate between ``start_s`` and ``stop_s``.

    Deceleration stops when the lead reaches standstill; the plant clamps speed
    at zero, so ``decel_mps2`` is applied only while the lead is still moving.
    """

    def fn(t_s: float, lead_v: float, ego_v: float, gap_m: float) -> float:
        if start_s - 1e-9 <= t_s < stop_s and lead_v > 0.0:
            return -abs(decel_mps2)
        return 0.0

    return fn


def lead_accelerates(accel_mps2: float, cap_mps: float) -> LeadAccelFn:
    """The lead pulls away at ``accel_mps2`` until it reaches ``cap_mps``."""

    def fn(t_s: float, lead_v: float, ego_v: float, gap_m: float) -> float:
        return abs(accel_mps2) if lead_v < cap_mps else 0.0

    return fn


def lead_brake_then_release(
    decel_mps2: float, start_s: float, duration_s: float, resume_accel_mps2: float, resume_to_mps: float
) -> LeadAccelFn:
    """Brake for a while, then accelerate back up.  The recovery test case."""

    def fn(t_s: float, lead_v: float, ego_v: float, gap_m: float) -> float:
        if start_s - 1e-9 <= t_s < start_s + duration_s and lead_v > 0.0:
            return -abs(decel_mps2)
        if t_s >= start_s + duration_s and lead_v < resume_to_mps:
            return abs(resume_accel_mps2)
        return 0.0

    return fn


def lateral_cut_in(
    from_m: float, to_m: float, start_s: float, duration_s: float
) -> LateralFn:
    """A lateral lane change: hold ``from_m``, slide to ``to_m``, hold.

    The slide is linear in time.  A real lane change is closer to a sine, but
    the quantity the in-path gate reacts to is the moment the footprints start
    to overlap, and linear and sinusoidal profiles cross that boundary within a
    frame of each other at any realistic duration.

    Args:
        from_m, to_m: Lateral positions, metres right of the true lane centre.
        start_s: When the manoeuvre begins.
        duration_s: How long it takes.  A comfortable motorway lane change is
            3-5 s; an aggressive cut-in is 1.5 s.
    """
    span = max(1e-9, float(duration_s))

    def fn(t_s: float) -> float:
        if t_s <= start_s:
            return float(from_m)
        if t_s >= start_s + span:
            return float(to_m)
        frac = (t_s - start_s) / span
        return float(from_m) + (float(to_m) - float(from_m)) * frac

    return fn


def adjacent_lane_object(
    initial_gap_m: float,
    speed_mps: float,
    accel_fn: Optional[LeadAccelFn] = None,
    side: float = 1.0,
    label: str = "car (next lane)",
    **kwargs: object,
) -> LeadSpec:
    """A vehicle one lane over: detected, geometrically real, and NOT to brake for.

    Placed at exactly one lane width from the true lane centre, so its footprint
    clears the ego corridor by ``LANE_WIDTH_M - LANE_HALF_WIDTH_M - width/2``
    = 0.85 m -- close enough that a lane estimate wrong by a metre puts it back
    inside, which is the point of :attr:`PerceptionSpec.lane_offset_error_m`.

    Args:
        side: ``+1`` for the lane to the right, ``-1`` for the lane to the left.
    """
    return LeadSpec(
        initial_gap_m=float(initial_gap_m),
        initial_speed_mps=float(speed_mps),
        accel_fn=accel_fn or lead_constant_speed(),
        lateral_offset_m=float(side) * LANE_WIDTH_M,
        label=label,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# The road
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RoadSpec:
    """Road geometry ahead of the ego.

    Attributes:
        curvature_1pm: Signed curvature (1/radius) as a function of time,
            positive for a left-hand bend.  A constant 1/230 is a 230 m radius
            bend, which at 20 m/s is 1.74 m/s^2 of lateral acceleration -- a
            brisk but entirely ordinary motorway curve.
        label: Human-readable description.
    """

    curvature_1pm: Callable[[float], float] = lambda t_s: 0.0
    label: str = "straight"


def straight_road() -> RoadSpec:
    return RoadSpec(curvature_1pm=lambda t_s: 0.0, label="straight")


def constant_bend(radius_m: float, sign: float = 1.0) -> RoadSpec:
    """A bend of constant radius.  ``sign`` is +1 for left, -1 for right."""
    kappa = sign / float(radius_m)
    return RoadSpec(
        curvature_1pm=lambda t_s: kappa,
        label="bend R=%.0f m (a_lat=%.2f m/s^2 at 20 m/s)" % (radius_m, 400.0 / radius_m),
    )


# --------------------------------------------------------------------------- #
# World state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObjectState:
    """The true state of one object at one frame.

    Ground truth in every field, including the two in-path answers.  The system
    never sees this; :class:`Sensor` decides what it is told.

    Attributes:
        index: Stable index of the object within the scenario.  0 is the lead
            when there is one.
        label: The spec's label.
        is_lead: Whether this is the scenario's designated lead.
        present: Whether the object is in the scene this frame.
        x_m: Absolute longitudinal position on the ego's axis, metres.
        gap_m: Clear longitudinal distance from the ego's front bumper, metres.
            Negative means the object is behind the ego.
        v_mps, a_mps2: True speed and acceleration.
        closing_mps: ``ego_v - object_v``; positive when the gap is shrinking.
        lateral_m: Lateral position, metres right of the TRUE lane centre.
        half_width_m: Half the object's width, metres.
        in_ego_lane: TRUE lane occupancy -- does the object's footprint overlap
            the ego's lane?  ``|lateral| <= LANE_HALF_WIDTH_M + half_width``.
            This is the answer the perception stack is *supposed* to produce.
        in_ego_path: TRUE collision-course overlap -- does the object's
            footprint overlap the ego's own footprint, wherever the ego has
            drifted to?  ``|lateral - ego_lateral| <= EGO_HALF_WIDTH_M +
            half_width``.  Narrower than ``in_ego_lane``, and it is this one
            that decides whether a collision is geometrically possible.
    """

    index: int
    label: str
    is_lead: bool
    present: bool
    x_m: float
    gap_m: float
    v_mps: float
    a_mps2: float
    closing_mps: float
    lateral_m: float
    half_width_m: float
    height_m: float
    in_ego_lane: bool
    in_ego_path: bool


@dataclass
class WorldState:
    """The complete true state of the world at one frame.

    Every field is ground truth.  The system under test never sees this object;
    it sees whatever :class:`Sensor` chooses to report about it.
    """

    frame: int = 0
    t_s: float = 0.0

    ego_x_m: float = 0.0
    ego_v_mps: float = 0.0
    ego_a_mps2: float = 0.0

    lead_present: bool = False
    lead_x_m: float = float("inf")
    lead_v_mps: float = 0.0
    lead_a_mps2: float = 0.0

    lateral_offset_m: float = 0.0
    """Ego offset from the lane centre, metres, positive to the right."""

    heading_err_rad: float = 0.0
    """Ego heading relative to the lane tangent, radians, positive to the left."""

    road_wheel_rad: float = 0.0
    """Road-wheel angle, radians, positive to the RIGHT -- the same sign
    convention as ``ControlCommand.steering``."""

    road_curvature_1pm: float = 0.0

    throttle: float = 0.0
    brake: float = 0.0
    steering: float = 0.0
    """The pedals and steering ACTUALLY APPLIED over the step that produced this
    state.  With ``PlantConfig.actuation_latency_s > 0`` these are not the
    commands issued this frame; see ``commanded_throttle`` and friends."""

    dt_s: float = DT_S
    """Duration of the step that produced this state, seconds.  At frame 0 this
    is the nominal period, because no step has happened yet to measure.  This is what a
    real pipeline measures as "time since the previous frame" and feeds to its
    controllers, and it is what :attr:`Observation.dt_s` reports."""

    objects: Tuple[ObjectState, ...] = ()
    """Every object in the world, lead first when there is one."""

    commanded_throttle: float = 0.0
    commanded_brake: float = 0.0
    commanded_steering: float = 0.0
    """The command ISSUED this frame, before the actuation delay and before
    brake override.  This is what the system asked for."""

    pedal_conflict: bool = False
    """The issued command asked for throttle AND brake at the same time.

    Recorded, never netted.  The plant does not decide whether this is a
    failure -- the judgement layer does -- but it refuses to hide it.
    """

    throttle_cut_by_brake: bool = False
    """Brake override fired on the applied command: a non-zero throttle request
    was discarded because the brake was down."""

    accel_demand_mps2: float = 0.0
    """The acceleration the applied pedals demanded, before actuator lag."""

    @property
    def gap_m(self) -> float:
        """True bumper-to-bumper clear distance to the LEAD, or +inf when there
        is no lead.  Objects other than the lead are in :attr:`objects`."""
        if not self.lead_present:
            return float("inf")
        return self.lead_x_m - self.ego_x_m

    @property
    def closing_mps(self) -> float:
        """True closing rate on the lead: positive when the gap is shrinking."""
        if not self.lead_present:
            return 0.0
        return self.ego_v_mps - self.lead_v_mps

    @property
    def in_path_objects(self) -> Tuple[ObjectState, ...]:
        """Present objects ahead of the ego whose footprint overlaps its path."""
        return tuple(
            o for o in self.objects if o.present and o.gap_m > 0.0 and o.in_ego_path
        )

    @property
    def min_in_path_gap_m(self) -> float:
        """Smallest forward gap over every object actually in the ego's path.

        This, not :attr:`gap_m`, is the quantity a multi-object collision check
        wants: the nearest thing the ego can hit.
        """
        gaps = [o.gap_m for o in self.in_path_objects]
        return min(gaps) if gaps else float("inf")


def _in_lane(lateral_m: float, half_width_m: float) -> bool:
    """Footprint overlap between an object and the ego's lane."""
    return abs(lateral_m) <= LANE_HALF_WIDTH_M + half_width_m


def _in_path(lateral_m: float, half_width_m: float, ego_lateral_m: float) -> bool:
    """Footprint overlap between an object and the ego's own swept corridor."""
    return abs(lateral_m - ego_lateral_m) <= EGO_HALF_WIDTH_M + half_width_m


# --------------------------------------------------------------------------- #
# The plant
# --------------------------------------------------------------------------- #


class Plant:
    """A deterministic vehicle-and-world simulator.

    Usage::

        plant = Plant(ego_speed_mps=20.0, lead=lead_spec, road=road_spec)
        state = plant.state              # frame 0, before any command
        while ...:
            state = plant.step(command)  # advance one dt with that command

    ``step`` applies the command through the actuator lags, integrates one
    timestep and returns the new true state.  It is pure with respect to
    everything outside the instance: no clock, no global RNG.

    Args:
        ego_speed_mps: Ego speed at t=0.
        lead: The designated lead, or None for no lead.  The oracle's
            counterfactual kinematics are computed against this object.
        road: Road geometry.
        config: Vehicle and integration parameters.
        initial_lateral_offset_m: Ego offset from the lane centre at t=0.
        others: Any number of additional objects.  They are simulated exactly
            like the lead and reported exactly like it; the only thing that
            makes the lead special is that the oracle judges avoidability
            against it.
    """

    def __init__(
        self,
        ego_speed_mps: float,
        lead: Optional[LeadSpec] = None,
        road: Optional[RoadSpec] = None,
        config: PlantConfig = DEFAULT_PLANT,
        initial_lateral_offset_m: float = 0.0,
        others: Sequence[LeadSpec] = (),
    ) -> None:
        self.config = config
        self.lead = lead
        self.road = road or straight_road()

        self._specs: List[LeadSpec] = []
        if lead is not None:
            self._specs.append(lead)
        self._lead_index: Optional[int] = 0 if lead is not None else None
        self._specs.extend(others)

        self._obj_x = [float(s.initial_gap_m) for s in self._specs]
        self._obj_v = [float(s.initial_speed_mps) for s in self._specs]
        self._obj_a = [0.0] * len(self._specs)

        self._alpha_cache: Dict[Tuple[float, float], float] = {}
        self._cmd_queue: List[Tuple[float, float, float]] = []
        self.pedal_conflict_frames: List[int] = []
        """Frames on which the ISSUED command asked for throttle and brake at
        once.  Exposed for the judgement layer; the plant asserts nothing."""

        # The objects' absolute positions are measured on the same axis as the
        # ego's, so each starts at its initial gap because ego_x_m starts at 0.
        self._state = self._compose(
            frame=0,
            t_s=0.0,
            dt_s=config.dt_s,
            ego_x_m=0.0,
            ego_v_mps=float(ego_speed_mps),
            ego_a_mps2=0.0,
            lateral_offset_m=float(initial_lateral_offset_m),
            heading_err_rad=0.0,
            road_wheel_rad=0.0,
            road_curvature_1pm=self.road.curvature_1pm(0.0),
            applied=(0.0, 0.0, 0.0),
            commanded=(0.0, 0.0, 0.0),
            pedal_conflict=False,
            throttle_cut=False,
            accel_demand=0.0,
        )

    # -------------------------------------------------------------- helpers

    @property
    def actuation_delay_frames(self) -> int:
        """Command dead time expressed in whole frames of the nominal period."""
        if self.config.actuation_latency_s <= 0.0 or self.config.dt_s <= 0.0:
            return 0
        return int(round(self.config.actuation_latency_s / self.config.dt_s))

    def dt_for_frame(self, frame: int) -> float:
        """Duration of the step that advances the world from ``frame``."""
        return self.config.dt_for_frame(frame)

    @property
    def next_dt_s(self) -> float:
        """Duration of the step the next :meth:`step` call will take.

        A closed loop that wants to tell the system how long its own frame was
        should pass ``state.dt_s`` (elapsed since the previous frame, which is
        what a real pipeline measures); this property is the *future* interval,
        which no real system knows in advance.
        """
        return self.dt_for_frame(self._state.frame)

    def _alpha(self, dt_s: float, rise_time_s: float) -> float:
        key = (dt_s, rise_time_s)
        got = self._alpha_cache.get(key)
        if got is None:
            got = _lag_alpha(dt_s, rise_time_s)
            self._alpha_cache[key] = got
        return got

    def _compose(
        self,
        frame: int,
        t_s: float,
        dt_s: float,
        ego_x_m: float,
        ego_v_mps: float,
        ego_a_mps2: float,
        lateral_offset_m: float,
        heading_err_rad: float,
        road_wheel_rad: float,
        road_curvature_1pm: float,
        applied: Tuple[float, float, float],
        commanded: Tuple[float, float, float],
        pedal_conflict: bool,
        throttle_cut: bool,
        accel_demand: float,
    ) -> WorldState:
        """Build a :class:`WorldState` including every object's derived truth."""
        objects: List[ObjectState] = []
        for i, spec in enumerate(self._specs):
            present = spec.present_at(t_s)
            lateral = spec.lateral_at(t_s)
            half_w = 0.5 * float(spec.width_m)
            gap = self._obj_x[i] - ego_x_m
            objects.append(
                ObjectState(
                    index=i,
                    label=spec.label or "object %d" % i,
                    is_lead=(i == self._lead_index),
                    present=present,
                    x_m=self._obj_x[i],
                    gap_m=gap,
                    v_mps=self._obj_v[i],
                    a_mps2=self._obj_a[i],
                    closing_mps=ego_v_mps - self._obj_v[i],
                    lateral_m=lateral,
                    half_width_m=half_w,
                    height_m=float(spec.height_m),
                    in_ego_lane=present and _in_lane(lateral, half_w),
                    in_ego_path=present and _in_path(lateral, half_w, lateral_offset_m),
                )
            )

        li = self._lead_index
        return WorldState(
            frame=frame,
            t_s=t_s,
            ego_x_m=ego_x_m,
            ego_v_mps=ego_v_mps,
            ego_a_mps2=ego_a_mps2,
            lead_present=objects[li].present if li is not None else False,
            lead_x_m=objects[li].x_m if li is not None else float("inf"),
            lead_v_mps=objects[li].v_mps if li is not None else 0.0,
            lead_a_mps2=objects[li].a_mps2 if li is not None else 0.0,
            lateral_offset_m=lateral_offset_m,
            heading_err_rad=heading_err_rad,
            road_wheel_rad=road_wheel_rad,
            road_curvature_1pm=road_curvature_1pm,
            throttle=applied[0],
            brake=applied[1],
            steering=applied[2],
            dt_s=dt_s,
            objects=tuple(objects),
            commanded_throttle=commanded[0],
            commanded_brake=commanded[1],
            commanded_steering=commanded[2],
            pedal_conflict=pedal_conflict,
            throttle_cut_by_brake=throttle_cut,
            accel_demand_mps2=accel_demand,
        )

    @classmethod
    def from_state(
        cls,
        state: WorldState,
        lead: Optional[LeadSpec] = None,
        road: Optional[RoadSpec] = None,
        config: PlantConfig = DEFAULT_PLANT,
        others: Sequence[LeadSpec] = (),
    ) -> "Plant":
        """A plant resumed from an existing true state.

        Used by the oracle to ask counterfactual questions ("what if the ego had
        braked flat out from here?") against the same physics the run itself
        used, including the actuator lag already built up.

        Only the objects passed here are resumed.  The oracle asks its question
        about the lead, so it passes the lead; a caller that wants the other
        traffic back must pass ``others`` too.  Object positions are taken from
        ``state`` so the resumed world starts exactly where the real one was.
        """
        plant = cls(
            ego_speed_mps=state.ego_v_mps,
            lead=lead,
            road=road,
            config=config,
            others=others,
        )
        for i, obj in enumerate(state.objects[: len(plant._specs)]):
            plant._obj_x[i] = obj.x_m
            plant._obj_v[i] = obj.v_mps
            plant._obj_a[i] = obj.a_mps2
        if lead is not None and state.lead_present and not state.objects:
            plant._obj_x[0] = state.lead_x_m
            plant._obj_v[0] = state.lead_v_mps
            plant._obj_a[0] = state.lead_a_mps2
        plant._state = plant._compose(
            frame=state.frame,
            t_s=state.t_s,
            dt_s=state.dt_s,
            ego_x_m=state.ego_x_m,
            ego_v_mps=state.ego_v_mps,
            ego_a_mps2=state.ego_a_mps2,
            lateral_offset_m=state.lateral_offset_m,
            heading_err_rad=state.heading_err_rad,
            road_wheel_rad=state.road_wheel_rad,
            road_curvature_1pm=state.road_curvature_1pm,
            applied=(state.throttle, state.brake, state.steering),
            commanded=(
                state.commanded_throttle,
                state.commanded_brake,
                state.commanded_steering,
            ),
            pedal_conflict=state.pedal_conflict,
            throttle_cut=state.throttle_cut_by_brake,
            accel_demand=state.accel_demand_mps2,
        )
        return plant

    @property
    def state(self) -> WorldState:
        """The current true state.  Do not mutate."""
        return self._state

    # ---------------------------------------------------------------- stepping

    def step(
        self,
        throttle: float,
        brake: float,
        steering: float,
        dt_s: Optional[float] = None,
    ) -> WorldState:
        """Advance the world by one timestep under the given actuator command.

        Args:
            throttle: Pedal fraction in [0, 1].
            brake: Pedal fraction in [0, 1].
            steering: Normalised steering in [-1, 1]; 1.0 is
                :data:`MAX_ROAD_WHEEL_RAD` at the road wheel.
            dt_s: Override the step duration.  Normally None, in which case the
                duration comes from :attr:`PlantConfig.dt_schedule` (falling back
                to the nominal period), so a scenario can inject a dropped or
                overrunning frame without the caller doing anything.

        Returns:
            The new :class:`WorldState`.
        """
        cfg = self.config
        s = self._state
        dt = float(dt_s) if dt_s is not None else self.dt_for_frame(s.frame)

        c_thr = min(1.0, max(0.0, float(throttle)))
        c_brk = min(1.0, max(0.0, float(brake)))
        c_str = min(1.0, max(-1.0, float(steering)))

        # --- throttle/brake conflict, recorded and NOT netted ---------------- #
        conflict = c_thr > 0.0 and c_brk > 0.0
        if conflict:
            self.pedal_conflict_frames.append(s.frame)

        # --- actuation dead time -------------------------------------------- #
        self._cmd_queue.append((c_thr, c_brk, c_str))
        depth = self.actuation_delay_frames
        if len(self._cmd_queue) > depth:
            a_thr, a_brk, a_str = self._cmd_queue.pop(0)
        else:
            a_thr, a_brk, a_str = 0.0, 0.0, 0.0

        # --- longitudinal ---------------------------------------------------- #
        # Brake override: at or above the pedal-switch threshold the throttle
        # request is discarded outright rather than subtracted.  A real car does
        # this, and it is what makes a simultaneous throttle-and-brake command
        # observable instead of arithmetically absorbed.
        throttle_cut = a_brk >= cfg.brake_override_pedal and a_thr > 0.0
        if a_brk >= cfg.brake_override_pedal:
            demand = -cfg.max_brake_decel_mps2 * a_brk
        else:
            demand = cfg.max_accel_mps2 * a_thr - cfg.max_brake_decel_mps2 * a_brk
        demand = min(cfg.max_accel_mps2, max(-cfg.max_brake_decel_mps2, demand))

        # Lag: use the brake time constant when the demand is a deceleration,
        # the throttle one otherwise.  Braking builds faster than torque does.
        rise = cfg.brake_rise_time_s if demand < s.ego_a_mps2 else cfg.throttle_rise_time_s
        alpha = self._alpha(dt, rise)
        a_ego = s.ego_a_mps2 + (demand - s.ego_a_mps2) * alpha

        v_new = s.ego_v_mps + a_ego * dt
        if v_new < 0.0:
            # The car cannot reverse under braking.  Truncate the step and
            # record the acceleration actually realised over it.
            a_ego = -s.ego_v_mps / dt if dt > 0 else 0.0
            v_new = 0.0
        # Trapezoidal integration, which is exact for a constant acceleration
        # over the step.
        ego_x = s.ego_x_m + 0.5 * (s.ego_v_mps + v_new) * dt

        # --- objects ---------------------------------------------------------- #
        t_next = s.t_s + dt
        for i, spec in enumerate(self._specs):
            gap_i = self._obj_x[i] - s.ego_x_m
            a_obj = float(spec.accel_fn(s.t_s, self._obj_v[i], s.ego_v_mps, gap_i))
            v_obj = self._obj_v[i] + a_obj * dt
            if v_obj < 0.0:
                a_obj = -self._obj_v[i] / dt if dt > 0 else 0.0
                v_obj = 0.0
            self._obj_x[i] = self._obj_x[i] + 0.5 * (self._obj_v[i] + v_obj) * dt
            self._obj_v[i] = v_obj
            self._obj_a[i] = a_obj

        # --- lateral ---------------------------------------------------------- #
        # Bicycle model, small-angle.  The road turns under the ego at
        # v * kappa rad/s; the ego turns at v * tan(delta) / L rad/s.  The
        # difference is the heading error rate, and the heading error carries
        # the ego across the lane at v * sin(heading) m/s.
        # ``steering`` and ``road_wheel_rad`` are POSITIVE TO THE RIGHT, matching
        # the control command's own convention (adas.planning.lateral: a positive
        # steering angle means the lane centre is to the right, so steer right).
        # The bicycle model below works in absolute headings, which are positive
        # to the LEFT, hence the single negation.
        delta_demand = a_str * cfg.max_road_wheel_rad
        steer_alpha = self._alpha(dt, cfg.steering_rise_time_s)
        delta = s.road_wheel_rad + (delta_demand - s.road_wheel_rad) * steer_alpha
        kappa = float(self.road.curvature_1pm(t_next))
        v_mid = 0.5 * (s.ego_v_mps + v_new)
        yaw_rate = -v_mid * math.tan(delta) / cfg.wheelbase_m
        heading = s.heading_err_rad + (yaw_rate - v_mid * kappa) * dt
        lateral = s.lateral_offset_m - v_mid * math.sin(heading) * dt

        self._state = self._compose(
            frame=s.frame + 1,
            t_s=t_next,
            dt_s=dt,
            ego_x_m=ego_x,
            ego_v_mps=v_new,
            ego_a_mps2=a_ego,
            lateral_offset_m=lateral,
            heading_err_rad=heading,
            road_wheel_rad=delta,
            road_curvature_1pm=kappa,
            applied=(a_thr, a_brk, a_str),
            commanded=(c_thr, c_brk, c_str),
            pedal_conflict=conflict,
            throttle_cut=throttle_cut,
            accel_demand=demand,
        )
        return self._state


# --------------------------------------------------------------------------- #
# Perception model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PerceptionSpec:
    """How the (simulated) perception stack reports the true world.

    Everything here is a *sensor* characteristic.  It never changes the world,
    only what the system is told about it, which is what makes the difference
    between "the system was wrong" and "the system was lied to" measurable.

    Attributes:
        max_range_m: Beyond this true range an object is not reported at all.
            80 m is a realistic detection horizon for a 1280x720 mono camera
            against a car-sized target.
        max_lateral_fov_m: Objects further than this from the ego's own
            centreline are outside the camera's horizontal field of view and are
            not reported.  8 m at any useful range is roughly a 60 degree lens's
            coverage of two lanes either side.
        range_noise_m: Standard deviation of zero-mean Gaussian range noise,
            metres.  Drawn from the counter-based stream, so it is reproducible
            and independent of scene composition.  **The arbiter's corroboration
            logic exists because of a false positive measured at +/-0.30 m of
            range noise**; a harness with no noise cannot reproduce it, which is
            why :func:`noisy_perception` defaults to exactly that figure.
        range_bias_frac: Systematic multiplicative range error (0.1 = 10% far).
            A mono camera's range is ``f*H/h``: get the height prior H wrong by
            10% -- a saloon prior applied to an SUV -- and every range is 10%
            wrong, in the same direction, forever.  Bias does not average out
            and a corroboration window cannot see it.
        lateral_noise_m: Standard deviation on the reported lateral offset.
        box_noise_px: Standard deviation of box-corner jitter, pixels.  Only
            meaningful with ``use_real_tracker``, where range is derived from
            the box: one pixel of jitter on a 30 m car is about 0.6 m of range.
        miss_frames: Frame indices on which the detector produces nothing at
            all, while perception itself still reports healthy.  This is a
            detection miss, not a sensor failure.
        failed_frames: Frame indices on which perception reports unhealthy
            (``PerceptionStatus.ok = False``).  This is a sensor/pipeline
            failure; the track list is empty AND the system is told why.
        source_lost_from_frame: From this frame onward perception is failed for
            the rest of the run.  Models a camera that stops delivering.
        reid_frames: Frame indices on which the tracker assigns a NEW track id
            to the same physical vehicle.  Everything downstream sees a
            brand-new object with no history -- including no range history, so
            no measurable closing rate.  See :attr:`rate_window_frames`.
        range_jump_at: ``(frame, delta_m)`` -- a one-off step in the reported
            range of the LEAD at that frame, persisting afterwards.  Models a
            range re-estimate after a re-identification.
        lane_center_bias_px: Constant error added to the reported lane-centre
            column.  A large value models a mock/failed lane estimate.
        lane_offset_error_m: Metric error in the lane estimate: the estimator
            believes the lane centre is this many metres to the RIGHT of where
            it really is.  It shifts the reported lane-centre column, the
            reported ego lateral offset and -- the point -- the reported
            ``in_ego_lane`` flag, all consistently.  With a next-lane vehicle at
            3.5 m and a 1.8 m car, an error of 0.86 m or more drags that vehicle
            inside the reported ego lane, which is precisely the round-1
            lane-error blindness.
        lane_is_mock: Whether the reported lane model is flagged as mock.
        lane_confidence: Confidence reported with the lane model.
        camera_calibrated: Whether the camera has a usable homography.  One
            flag, because one homography produces all of it: with it the lane
            model carries metric ground-plane boundary polynomials (which is
            what the tracker's own ego-lane corridor test reads) and the ego's
            metric lateral offset is published; without it the lane is a
            pixel-space column and nothing metric is available, which is the
            state README.md records this board in by default ("Real,
            UNCALIBRATED by default").  True here because the harness's closed
            loop already asserts a calibrated camera to the arbiter; a scenario
            that wants to test the uncalibrated path sets it False.
        sense_latency_s: Capture-to-command delay; see :data:`SENSE_LATENCY_S`.
            Rounded to the nearest whole frame against the actual elapsed times
            in the run, so a frame overrun stretches the delay exactly as it
            does on the road.
        rate_window_frames: Number of past range measurements used to estimate
            the closing rate.  A camera measures range, never range rate; the
            reported ``velocity_mps`` is the negated least-squares slope of the
            track's own measured range history over this window.  5 frames is
            0.25 s, the same order as the production tracker's Kalman range
            filter settling time.
        rate_min_samples: Fewest samples that will produce a rate at all.  Below
            it the track reports ``velocity_mps = 0.0`` and ``ttc_s = inf``,
            because it genuinely does not know.  This is what makes a
            re-identification scenario mean something: the old model reset the
            track id and then handed the new track the TRUE closing rate, so the
            defect it was named after could not occur.
        ego_speed_noise_mps: Noise on the reported ego speed.
        use_real_tracker: Run detections through
            :class:`adas.tracking.MultiObjectTracker` instead of injecting
            :class:`TrackedObject` directly.  Off by default: injected tracks
            are fast and isolate the decision layer.  On, the Kalman range
            filter, Hungarian association, M-of-N confirmation and the tracker's
            own ``in_ego_lane`` decision all run -- and the phantom AEB lived
            exactly there.  Costs three frames of confirmation latency on every
            new object (README item 12: up to 150 ms), which is real.
        tracker_confirm_hits, tracker_confirm_window, tracker_max_missed:
            Passed straight to the tracker when ``use_real_tracker``.
        seed: Seed for the noise streams.  Fixed, so runs are reproducible.
    """

    max_range_m: float = 80.0
    max_lateral_fov_m: float = 8.0
    range_noise_m: float = 0.0
    range_bias_frac: float = 0.0
    lateral_noise_m: float = 0.0
    box_noise_px: float = 0.0
    miss_frames: Tuple[int, ...] = ()
    failed_frames: Tuple[int, ...] = ()
    source_lost_from_frame: Optional[int] = None
    reid_frames: Tuple[int, ...] = ()
    range_jump_at: Optional[Tuple[int, float]] = None
    lane_center_bias_px: float = 0.0
    lane_offset_error_m: float = 0.0
    lane_is_mock: bool = False
    lane_confidence: float = 0.85
    camera_calibrated: bool = True
    sense_latency_s: float = SENSE_LATENCY_S
    rate_window_frames: int = 5
    rate_min_samples: int = 2
    ego_speed_noise_mps: float = 0.0
    use_real_tracker: bool = False
    tracker_confirm_hits: int = 3
    tracker_confirm_window: int = 5
    tracker_max_missed: int = 5
    seed: int = 20240914


PERFECT_PERCEPTION = PerceptionSpec()
"""Noise-free, bias-free perception -- but NOT instantaneous.

"Perfect" here means the measurements are exact, not that they arrive for free.
The one-frame :data:`SENSE_LATENCY_S` stays, because it is a property of the
computer, not of the sensor, and no amount of perception quality removes it.
"""

INSTANT_PERCEPTION = PerceptionSpec(sense_latency_s=0.0)
"""Exact AND instantaneous.  Physically impossible; use it only to isolate a
decision-layer question from the latency, and say so in the scenario's physics.
"""


def noisy_perception(
    range_noise_m: float = 0.30,
    range_bias_frac: float = 0.0,
    seed: int = 20240914,
    **kwargs: object,
) -> PerceptionSpec:
    """Perception with the historically significant noise already dialled in.

    The default of 0.30 m is not a round number chosen for looks: the arbiter's
    corroboration window was added in response to a false positive measured at
    +/-0.30 m of range noise, and until this harness could produce that noise it
    could not produce that false positive either.

    Args:
        range_noise_m: Gaussian range noise standard deviation, metres.
        range_bias_frac: Systematic multiplicative range error.
        seed: Noise seed.
        **kwargs: Any other :class:`PerceptionSpec` field.
    """
    return PerceptionSpec(
        range_noise_m=float(range_noise_m),
        range_bias_frac=float(range_bias_frac),
        seed=int(seed),
        **kwargs,  # type: ignore[arg-type]
    )


@dataclass
class Observation:
    """What the system under test is given for one frame.

    Attributes:
        frame: The DECISION frame index -- now.
        t_s: The decision time.
        ego: Ego speed, from the vehicle bus.  Not delayed: the CAN speed signal
            does not go through the vision pipeline.
        tracks: What perception reports.  These describe the world as it was at
            :attr:`measurement_t_s`, which is :attr:`sense_latency_s` earlier.
        perception: Health.
        lane_center_px, lane: The lane estimate, also delayed.
        dt_s: Time elapsed since the previous frame, seconds.  This is what a
            real pipeline measures and hands to its controllers, and it is NOT
            necessarily the nominal period; see
            :attr:`PlantConfig.dt_schedule`.
        lateral_offset_m: Metric ego offset from the lane centre, or None when
            the camera is uncalibrated (the default; see
            :attr:`PerceptionSpec.lateral_offset_available`).
        measurement_frame, measurement_t_s: Which true frame the measurements
            actually describe.  Ground truth about the *measurement*, not about
            the world; a consumer may use it to explain a decision but must not
            use it as a measurement.
        sense_latency_s: The realised delay this frame, seconds.
    """

    frame: int
    t_s: float
    ego: EgoState
    tracks: List[TrackedObject]
    perception: PerceptionStatus
    lane_center_px: Optional[float]
    lane: Optional[LaneModel]
    frame_width_px: int = FRAME_WIDTH_PX
    frame_height_px: int = FRAME_HEIGHT_PX
    lateral_offset_m: Optional[float] = None
    dt_s: float = DT_S
    measurement_frame: int = 0
    measurement_t_s: float = 0.0
    sense_latency_s: float = 0.0


def render_box(
    distance_m: float,
    lateral_offset_m: float = 0.0,
    width_m: float = CAR_WIDTH_M,
    height_m: float = CAR_HEIGHT_M,
    label: str = "car",
    confidence: float = 0.93,
) -> BoundingBox:
    """A geometrically consistent bounding box for an object at ``distance_m``.

    Flat-ground pinhole: the rear bumper contacts the road at
    ``horizon + f * h_cam / d`` and the roof is ``f * h_obj / d`` pixels above
    it.  The horizontal centre is the image centre displaced by the object's
    lateral offset relative to the EGO, ``f * y / d`` pixels.

    This exists so that the system's own in-path gate -- which works in image
    space -- sees a box that is consistent with the range it is also given.
    A harness that put a fixed box at a varying range would be testing a
    contradiction.

    Args:
        distance_m: Longitudinal range, metres.
        lateral_offset_m: Lateral offset from the EGO's centreline, positive
            right, metres.
        width_m, height_m: The object's true extent.  A 3.2 m truck rendered
            here and ranged by a 1.5 m height prior is mis-ranged by the same
            factor a real pinhole estimator gets wrong, which is the point of
            letting a scenario set it.
        label: Detection class label.
        confidence: Detection confidence.
    """
    d = max(1.0, float(distance_m))
    half_w_px = 0.5 * float(width_m) * CAMERA_FOCAL_PX / d
    h_px = float(height_m) * CAMERA_FOCAL_PX / d
    cx = FRAME_WIDTH_PX / 2.0 + lateral_offset_m * CAMERA_FOCAL_PX / d
    y2 = HORIZON_PX + CAMERA_HEIGHT_M * CAMERA_FOCAL_PX / d
    y2 = min(float(FRAME_HEIGHT_PX - 1), y2)
    y1 = max(0.0, y2 - h_px)
    return BoundingBox(cx - half_w_px, y1, cx + half_w_px, y2, confidence, label)


def lane_center_px_for(
    lateral_offset_m: float,
    curvature_1pm: float,
    lane_offset_error_m: float = 0.0,
) -> float:
    """The lane-centre column a lane detector would report.

    At a lookahead ``L`` the lane centre is displaced laterally by
    ``0.5 * kappa * L**2`` from the ego's heading, and the ego's own offset
    ``y`` moves it the other way.  Metres convert to pixels at ``f / L``.
    Positive curvature is a left-hand bend, which puts the lane centre left of
    the image centre, i.e. at a smaller column.

    Args:
        lateral_offset_m: The ego's TRUE offset from the lane centre, positive
            right.
        curvature_1pm: True road curvature.
        lane_offset_error_m: The estimator's metric error -- it believes the
            lane centre is this many metres to the right of the truth.  Applied
            here and, consistently, to the reported ``in_ego_lane`` flags, so
            that a scenario cannot fool the pixel gate and the flag by different
            amounts.
    """
    bend_m = 0.5 * curvature_1pm * LOOKAHEAD_M * LOOKAHEAD_M
    metres_right_of_ego = -bend_m - lateral_offset_m + float(lane_offset_error_m)
    return FRAME_WIDTH_PX / 2.0 + metres_right_of_ego * CAMERA_FOCAL_PX / LOOKAHEAD_M


class _RangeHistory:
    """Measured range samples for one track id, and the rate they imply.

    A mono camera measures range.  Every closing rate downstream of it is a
    difference of ranges over time, which is why range noise turns into rate
    noise amplified by ``1/dt`` and why a track with no history has no rate.
    Modelling that is the difference between a re-identification scenario that
    tests something and one that cannot fail.
    """

    __slots__ = ("_t", "_r", "_window")

    def __init__(self, window: int) -> None:
        self._t: List[float] = []
        self._r: List[float] = []
        self._window = max(2, int(window))

    def add(self, t_s: float, range_m: float) -> None:
        """Record one measurement, ignoring a repeat of a capture already held.

        A decision loop running faster than the camera sees the same image
        twice.  Storing it twice would put two points at the same abscissa and
        flatten the fitted slope towards zero -- a fabricated "it stopped
        closing", which is the exact signature this harness exists to catch.
        """
        if self._t and float(t_s) <= self._t[-1] + 1e-12:
            return
        self._t.append(float(t_s))
        self._r.append(float(range_m))
        if len(self._t) > self._window:
            del self._t[0]
            del self._r[0]

    @property
    def samples(self) -> int:
        return len(self._t)

    def closing_mps(self, min_samples: int) -> Optional[float]:
        """Least-squares closing rate, or None when there is not enough history.

        Positive when closing, matching ``TrackedObject.velocity_mps``: the
        slope of range against time is negative when the gap shrinks, so the
        reported rate is the negated slope.
        """
        n = len(self._t)
        if n < max(2, int(min_samples)):
            return None
        mean_t = sum(self._t) / n
        mean_r = sum(self._r) / n
        num = sum((self._t[i] - mean_t) * (self._r[i] - mean_r) for i in range(n))
        den = sum((self._t[i] - mean_t) ** 2 for i in range(n))
        if den <= 1e-12:
            return None
        return -(num / den)


class Sensor:
    """Turns true :class:`WorldState` into an :class:`Observation`.

    Stateful in exactly the ways a real perception stack is:

    * a **capture buffer**, because the answer it gives at time ``t`` was
      computed from an image taken at ``t - sense_latency``;
    * a **track-id counter** that changes on a re-identification;
    * a **per-track range history**, because closing rate is differentiated
      range and a new track has none;
    * a **dropout counter** feeding ``PerceptionStatus.consecutive_failures``.

    Noise is not part of that state: it comes from the counter-based stream in
    :func:`_gauss`, so it does not depend on call order or scene composition.

    Args:
        spec: The perception characteristics.
        config: The plant configuration, used only for the nominal frame period
            when resolving the latency to whole frames.
    """

    def __init__(
        self, spec: PerceptionSpec = PERFECT_PERCEPTION, config: PlantConfig = DEFAULT_PLANT
    ) -> None:
        self.spec = spec
        self.config = config
        self._buffer: List[WorldState] = []
        self._track_ids: Dict[int, int] = {}
        self._next_track_id = 1
        self._consecutive_failures = 0
        self._last_good_t_s = 0.0
        self._age: Dict[int, int] = {}
        self._hits: Dict[int, int] = {}
        self._history: Dict[int, _RangeHistory] = {}
        self._range_offset_m = 0.0
        self._tracker = None  # built lazily; see _real_tracker
        self._last_capture_frame: Optional[int] = None
        self._last_tracks: List[TrackedObject] = []
        nominal = config.dt_s if config.dt_s > 0 else DT_S
        self._buffer_depth = max(4, int(math.ceil(spec.sense_latency_s / nominal)) + 4)
        """Frames of capture history kept.  The delay needs
        ``ceil(latency/dt)`` of them; the rest is headroom for a schedule with
        frames shorter than nominal."""

    # ------------------------------------------------------------- internals

    def _delayed(self, now: WorldState) -> WorldState:
        """The buffered true state the current measurements describe.

        The capture instant is ``now.t_s - sense_latency_s``, quantised to the
        nearest frame actually in the buffer -- which is what a camera does: it
        cannot expose between frames.  Quantising by nearest rather than by
        floor keeps a 55 ms latency at exactly one frame on a 50 ms grid instead
        of rounding it up to two, and lets it stretch to two frames on its own
        once the frame period grows past 110 ms.

        Before the buffer is deep enough (the first frames of a run) the oldest
        available state is used.  A real pipeline's first command likewise acts
        on its first image, however old that is by the time it arrives.
        """
        lat = float(self.spec.sense_latency_s)
        if lat <= 0.0 or len(self._buffer) < 2:
            return self._buffer[-1]
        target = now.t_s - lat
        best = self._buffer[0]
        best_err = abs(best.t_s - target)
        for s in self._buffer:
            err = abs(s.t_s - target)
            if err <= best_err:
                best, best_err = s, err
        return best

    def _track_id_for(self, index: int) -> int:
        got = self._track_ids.get(index)
        if got is None:
            got = self._next_track_id
            self._next_track_id += 1
            self._track_ids[index] = got
        return got

    def _reidentify(self, index: int) -> None:
        """Give an object a brand-new identity, with no history of any kind."""
        old = self._track_ids.get(index)
        self._track_ids[index] = self._next_track_id
        self._next_track_id += 1
        self._age[index] = 0
        self._hits[index] = 0
        if old is not None:
            self._history.pop(old, None)
        self._history.pop(self._track_ids[index], None)

    def _real_tracker(self):
        """Build the production tracker on first use.

        Imported here, not at module scope, so that the synthetic path never
        pays for it and a machine without numpy can still import the plant.
        The tracker itself is CPU-only: numpy and nothing else.
        """
        if self._tracker is None:
            from adas.tracking import MultiObjectTracker

            self._tracker = MultiObjectTracker(
                focal_length_px=CAMERA_FOCAL_PX,
                object_height_m=CAR_HEIGHT_M,
                frame_width_px=FRAME_WIDTH_PX,
                frame_height_px=FRAME_HEIGHT_PX,
                confirm_hits=self.spec.tracker_confirm_hits,
                confirm_window=self.spec.tracker_confirm_window,
                max_missed=self.spec.tracker_max_missed,
                ego_half_width_m=EGO_HALF_WIDTH_M,
                nominal_lane_width_m=LANE_WIDTH_M,
            )
        return self._tracker

    # ------------------------------------------------------------------- API

    def observe(self, state: WorldState) -> Observation:
        """Report ``state`` through the configured perception characteristics.

        Args:
            state: The CURRENT true state.  The measurements returned describe
                an earlier one; see :meth:`_delayed`.

        Returns:
            The :class:`Observation` the system under test is given this frame.
        """
        spec = self.spec
        frame = state.frame
        self._buffer.append(state)
        # Enough history to resolve the configured latency at the nominal
        # period, with headroom for the shortest frames a schedule might use,
        # and no more -- a 10 000-frame run must not accumulate 10 000 states.
        while len(self._buffer) > self._buffer_depth:
            del self._buffer[0]

        measured_state = self._delayed(state)

        failed = frame in spec.failed_frames or (
            spec.source_lost_from_frame is not None and frame >= spec.source_lost_from_frame
        )

        if spec.range_jump_at is not None and frame == spec.range_jump_at[0]:
            self._range_offset_m += float(spec.range_jump_at[1])
        if frame in spec.reid_frames:
            for i in range(len(measured_state.objects)):
                self._reidentify(i)

        ego_speed = state.ego_v_mps + _gauss(
            spec.seed, frame, _STREAM_EGO_SPEED, 0, spec.ego_speed_noise_mps
        )
        ego = EgoState(speed_mps=max(0.0, ego_speed), valid=True, timestamp_s=state.t_s)

        if failed:
            self._consecutive_failures += 1
            # A failed frame publishes nothing, and nothing it published before
            # may be resurrected once it recovers: the cache is cleared so the
            # first healthy frame re-derives its tracks from a fresh capture.
            self._last_capture_frame = None
            self._last_tracks = []
            perception = PerceptionStatus(
                ok=False,
                consecutive_failures=self._consecutive_failures,
                last_good_timestamp_s=self._last_good_t_s,
                detector_ok=False,
                lane_ok=False,
                reason="source_lost" if spec.source_lost_from_frame is not None else "detector",
            )
            return Observation(
                frame=frame,
                t_s=state.t_s,
                ego=ego,
                tracks=[],
                perception=perception,
                lane_center_px=None,
                lane=None,
                lateral_offset_m=None,
                dt_s=state.dt_s,
                measurement_frame=measured_state.frame,
                measurement_t_s=measured_state.t_s,
                sense_latency_s=state.t_s - measured_state.t_s,
            )

        self._consecutive_failures = 0
        self._last_good_t_s = state.t_s
        perception = PerceptionStatus(ok=True, last_good_timestamp_s=state.t_s)

        centre = lane_center_px_for(
            measured_state.lateral_offset_m,
            measured_state.road_curvature_1pm,
            lane_offset_error_m=spec.lane_offset_error_m,
        )
        centre += spec.lane_center_bias_px
        lane = self._lane_model(measured_state, centre)

        # A detection miss is a property of the DECISION frame: the library says
        # "on frame N the system is handed nothing", and that is what happens,
        # whatever image frame N was reading.  The cache is invalidated so the
        # next frame re-derives its tracks rather than resurrecting these.
        if frame in spec.miss_frames:
            self._last_capture_frame = None
            self._last_tracks = []
            return Observation(
                frame=frame,
                t_s=state.t_s,
                ego=ego,
                tracks=[],
                perception=perception,
                lane_center_px=centre,
                lane=lane,
                lateral_offset_m=(
                    (measured_state.lateral_offset_m - spec.lane_offset_error_m)
                    if spec.camera_calibrated
                    else None
                ),
                dt_s=state.dt_s,
                measurement_frame=measured_state.frame,
                measurement_t_s=measured_state.t_s,
                sense_latency_s=state.t_s - measured_state.t_s,
            )

        # One image, one tracker update.  At startup, and whenever the decision
        # loop is faster than the capture it is reading, two consecutive frames
        # resolve to the same captured state; re-running the tracker on it would
        # double-count its age and hits and feed the Kalman filter a
        # zero-elapsed-time repeat.  A real pipeline publishes the previous
        # result instead, and so does this.
        if measured_state.frame == self._last_capture_frame:
            tracks = self._last_tracks
        elif spec.use_real_tracker:
            tracks = self._tracks_via_tracker(measured_state, state.dt_s, lane)
        else:
            tracks = self._tracks_injected(measured_state)
        self._last_capture_frame = measured_state.frame
        self._last_tracks = tracks

        reported_offset: Optional[float] = None
        if spec.camera_calibrated:
            reported_offset = (
                measured_state.lateral_offset_m
                - spec.lane_offset_error_m
                + _gauss(spec.seed, frame, _STREAM_LANE, 0, spec.lateral_noise_m)
            )

        return Observation(
            frame=frame,
            t_s=state.t_s,
            ego=ego,
            tracks=tracks,
            perception=perception,
            lane_center_px=centre,
            lane=lane,
            lateral_offset_m=reported_offset,
            dt_s=state.dt_s,
            measurement_frame=measured_state.frame,
            measurement_t_s=measured_state.t_s,
            sense_latency_s=state.t_s - measured_state.t_s,
        )

    # --------------------------------------------------------------- helpers

    def _lane_model(self, measured: WorldState, centre_px: float) -> LaneModel:
        """The lane estimate, in the pixel and metric forms a real fit produces.

        The metric half is not decoration.  ``MultiObjectTracker`` decides
        ``in_ego_lane`` from ``LaneModel.lines`` -- ground-plane polynomials
        ``x(z) = a z^2 + b z + c``, metres right of the ego at ``z`` metres
        ahead -- and falls back to a heuristic image band, 1.6x wider than a
        lane, when they are absent.  A harness that shipped an empty
        ``LaneModel`` therefore never tested the lane gate at all; it tested the
        fallback.

        The centreline is ``-0.5 kappa z^2 - y_ego + e``, where ``y_ego`` is the
        ego's true offset (positive right) and ``e`` is
        :attr:`PerceptionSpec.lane_offset_error_m`.  The quadratic term is the
        standard small-angle approximation to a constant-curvature arc, and it
        is the SAME expression :func:`lane_center_px_for` evaluates at the
        20 m lookahead, so the pixel column and the metric polynomial can never
        disagree about where the estimator thinks the lane is.
        """
        spec = self.spec
        kappa = measured.road_curvature_1pm
        a = -0.5 * kappa
        c = -measured.lateral_offset_m + spec.lane_offset_error_m
        left = (a, 0.0, c - LANE_HALF_WIDTH_M)
        right = (a, 0.0, c + LANE_HALF_WIDTH_M)
        lines: List[LaneLine] = []
        if spec.camera_calibrated:
            lines = [
                LaneLine(points_px=[], coeffs=left, confidence=spec.lane_confidence, index=1),
                LaneLine(points_px=[], coeffs=right, confidence=spec.lane_confidence, index=2),
                LaneLine(
                    points_px=[], coeffs=(a, 0.0, c), confidence=spec.lane_confidence, index=-1
                ),
            ]
        return LaneModel(
            left_coeffs=left if spec.camera_calibrated else (0.0, 0.0, 0.0),
            right_coeffs=right if spec.camera_calibrated else (0.0, 0.0, 0.0),
            lane_center_px=centre_px,
            curvature_m=(1.0 / kappa) if kappa else 0.0,
            confidence=spec.lane_confidence,
            is_mock=spec.lane_is_mock,
            lines=lines,
        )

    def _detectable(self, measured: WorldState, frame: int) -> List[ObjectState]:
        """Objects the detector produces a box for this frame.

        Detection is a question about the camera and the object, never about the
        lane: an out-of-lane vehicle IS detected.  Deciding it is not a hazard
        is the system's job, and a harness that hid it would be doing that job
        for the system.

        Whole-frame detection misses are handled by the caller, against the
        decision frame; this method answers only "is it visible at all".
        """
        spec = self.spec
        out: List[ObjectState] = []
        for obj in measured.objects:
            if not obj.present:
                continue
            if obj.gap_m <= 0.0 or obj.gap_m > spec.max_range_m:
                continue
            if abs(obj.lateral_m - measured.lateral_offset_m) > spec.max_lateral_fov_m:
                continue
            out.append(obj)
        return out

    def _measured_range_m(self, obj: ObjectState, frame: int) -> float:
        """The range the stack reports for ``obj``: truth, biased, then noised.

        ``frame`` is the CAPTURE frame, not the decision frame, so one image
        yields one measurement however many decisions read it.
        """
        spec = self.spec
        measured = obj.gap_m * (1.0 + spec.range_bias_frac)
        if obj.is_lead:
            measured += self._range_offset_m
        measured += _gauss(spec.seed, frame, _STREAM_RANGE, obj.index, spec.range_noise_m)
        return max(0.3, measured)

    def _reported_lateral_m(self, obj: ObjectState, measured: WorldState, frame: int) -> float:
        """Object lateral offset relative to the EGO, as reported, with noise."""
        spec = self.spec
        rel = obj.lateral_m - measured.lateral_offset_m
        return rel + _gauss(spec.seed, frame, _STREAM_LATERAL, obj.index, spec.lateral_noise_m)

    def _tracks_injected(self, measured: WorldState) -> List[TrackedObject]:
        """Clean tracks, straight from the true state through the sensor model.

        Fast, and it isolates the decision layer: whatever goes wrong here went
        wrong in planning, control or arbitration, not in association.  What it
        does NOT skip is the honest parts of tracking -- identity, age, and the
        fact that closing rate is differentiated range.
        """
        spec = self.spec
        frame = measured.frame
        seen = {o.index for o in self._detectable(measured, frame)}
        tracks: List[TrackedObject] = []
        for obj in measured.objects:
            self._age[obj.index] = self._age.get(obj.index, 0) + 1
            if obj.index not in seen:
                continue
            tid = self._track_id_for(obj.index)
            self._hits[obj.index] = self._hits.get(obj.index, 0) + 1

            rng = self._measured_range_m(obj, frame)
            hist = self._history.setdefault(tid, _RangeHistory(spec.rate_window_frames))
            hist.add(measured.t_s, rng)
            rate = hist.closing_mps(spec.rate_min_samples)
            closing = 0.0 if rate is None else rate

            lateral = self._reported_lateral_m(obj, measured, frame)
            # The in-path flag the perception stack publishes, computed from the
            # lane centre it BELIEVES in.  With lane_offset_error_m = 0 this is
            # the truth; with an error it is confidently wrong, in the same
            # direction and by the same amount as the lane-centre column.
            believed_lateral_from_lane = obj.lateral_m - spec.lane_offset_error_m
            reported_in_lane = _in_lane(believed_lateral_from_lane, obj.half_width_m)

            ttc = rng / closing if closing > 1e-3 else float("inf")
            tracks.append(
                TrackedObject(
                    track_id=tid,
                    box=render_box(
                        rng,
                        lateral_offset_m=lateral,
                        width_m=2.0 * obj.half_width_m,
                        height_m=obj.height_m,
                        label=_label_of(obj),
                    ),
                    velocity_mps=closing,
                    distance_m=rng,
                    age_frames=self._age[obj.index],
                    hits=self._hits[obj.index],
                    time_since_update=0,
                    lateral_offset_m=lateral,
                    ttc_s=ttc,
                    in_ego_lane=reported_in_lane,
                )
            )
        return tracks

    def _tracks_via_tracker(
        self,
        measured: WorldState,
        dt_s: float,
        lane: LaneModel,
    ) -> List[TrackedObject]:
        """Detections through the real :class:`MultiObjectTracker`.

        Slow (a Hungarian solve per frame) and it withholds every object for its
        first ``confirm_hits`` frames, but it is the only path that exercises
        Kalman range filtering, data association, class-keyed height priors and
        the tracker's own ego-lane decision -- the code the phantom AEB lived
        in.  Range noise and bias are injected as an error on the rendered box's
        implied distance plus optional pixel jitter, because that is where a
        camera's range error physically comes from.
        """
        spec = self.spec
        frame = measured.frame
        detections: List[BoundingBox] = []
        for obj in self._detectable(measured, frame):
            rng = self._measured_range_m(obj, frame)
            lateral = self._reported_lateral_m(obj, measured, frame)
            box = render_box(
                rng,
                lateral_offset_m=lateral,
                width_m=2.0 * obj.half_width_m,
                height_m=obj.height_m,
                label=_label_of(obj),
            )
            jx = _gauss(spec.seed, frame, _STREAM_BOX_X, obj.index, spec.box_noise_px)
            jy = _gauss(spec.seed, frame, _STREAM_BOX_Y, obj.index, spec.box_noise_px)
            detections.append(
                BoundingBox(
                    box.x1 + jx,
                    box.y1 + jy,
                    box.x2 + jx,
                    box.y2 + jy,
                    box.confidence,
                    box.label,
                )
            )
        return list(
            self._real_tracker().update(
                detections,
                dt_s=float(dt_s),
                frame_width=FRAME_WIDTH_PX,
                frame_height=FRAME_HEIGHT_PX,
                lane=lane,
            )
        )


def _label_of(obj: ObjectState) -> str:
    """Detection class label for an object, defaulting to ``car``."""
    low = (obj.label or "").lower()
    for known in ("truck", "bus", "motorcycle", "bicycle", "person", "car"):
        if known in low:
            return known
    return "car"
